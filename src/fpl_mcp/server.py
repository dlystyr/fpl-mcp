from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Literal

import httpx
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
import uvicorn

logger = logging.getLogger("fpl-mcp")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

FPL_API_BASE = "https://fantasy.premierleague.com/api"
USER_AGENT = os.getenv("FPL_USER_AGENT", "fpl-mcp/1.0")

# Optional auth for public HTTPS endpoint
BEARER_TOKEN = os.getenv("MCP_BEARER_TOKEN", "")

Position = Literal["GKP", "DEF", "MID", "FWD"]
POS_MAP: dict[int, Position] = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

# Per-endpoint cache TTLs (seconds)
TTL_BOOTSTRAP = int(os.getenv("FPL_TTL_BOOTSTRAP", "120"))
TTL_FIXTURES = int(os.getenv("FPL_TTL_FIXTURES", "300"))
TTL_ELEMENT_SUMMARY = int(os.getenv("FPL_TTL_ELEMENT_SUMMARY", "1800"))

# Simple in-memory cache with per-key expiry
_cache: dict[str, tuple[float, Any]] = {}  # url -> (expires_at, data)


def _cache_get(url: str) -> Any | None:
    item = _cache.get(url)
    if not item:
        return None
    expires_at, data = item
    if time.time() >= expires_at:
        _cache.pop(url, None)
        return None
    return data


def _cache_set(url: str, data: Any, ttl: int) -> None:
    _cache[url] = (time.time() + max(1, ttl), data)


async def _get_json(path: str, ttl: int) -> Any:
    url = f"{FPL_API_BASE}/{path.lstrip('/')}"
    cached = _cache_get(url)
    if cached is not None:
        return cached

    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        data = r.json()

    _cache_set(url, data, ttl)
    return data


async def _bootstrap() -> dict[str, Any]:
    return await _get_json("bootstrap-static/", ttl=TTL_BOOTSTRAP)


async def _fixtures() -> list[dict[str, Any]]:
    return await _get_json("fixtures/", ttl=TTL_FIXTURES)


async def _element_summary(player_id: int) -> dict[str, Any]:
    return await _get_json(f"element-summary/{player_id}/", ttl=TTL_ELEMENT_SUMMARY)


def _price_m(now_cost: int) -> float:
    return now_cost / 10.0


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _current_event_id(events: list[dict[str, Any]]) -> int | None:
    cur = next((e for e in events if e.get("is_current")), None)
    if cur:
        return int(cur["id"])
    nxt = next((e for e in events if e.get("is_next")), None)
    if nxt:
        return int(nxt["id"])
    return None


def _fixture_difficulty_for_team(fx: dict[str, Any], team_id: int) -> int | None:
    if fx.get("team_h") == team_id:
        return int(fx.get("team_h_difficulty") or 0) or None
    if fx.get("team_a") == team_id:
        return int(fx.get("team_a_difficulty") or 0) or None
    return None


def _availability_penalty(el: dict[str, Any]) -> float:
    status = str(el.get("status", "a"))
    chance = el.get("chance_of_playing_next_round", None)
    penalty = 0.0
    if status != "a":
        penalty += 3.0
    if chance is not None:
        c = _to_float(chance, 100.0)
        if c < 75:
            penalty += 2.0
        if c < 50:
            penalty += 2.0
    return penalty


def _score_player_first_pass(
    el: dict[str, Any],
    teams_by_id: dict[int, dict[str, Any]],
    fixtures: list[dict[str, Any]],
    horizon_gws: int,
    current_event: int | None,
) -> dict[str, Any]:
    """
    Fast scoring using bootstrap + fixtures only.
    Used to select a candidate pool for second-pass refinement.
    """
    team_id = int(el["team"])
    pos = POS_MAP.get(int(el["element_type"]), "MID")
    price = _price_m(int(el["now_cost"]))

    ppg = _to_float(el.get("points_per_game"))
    form = _to_float(el.get("form"))
    ict = _to_float(el.get("ict_index"))
    minutes = int(_to_float(el.get("minutes")))

    diffs: list[int] = []
    if current_event is not None:
        for fx in fixtures:
            ev = fx.get("event")
            if ev is None:
                continue
            ev = int(ev)
            if ev < current_event or ev >= current_event + max(1, horizon_gws):
                continue
            d = _fixture_difficulty_for_team(fx, team_id)
            if d is not None:
                diffs.append(d)

    fixture_ease = (6.0 - (sum(diffs) / len(diffs))) if diffs else 0.0
    value = (ppg / price) if price else 0.0
    penalty = _availability_penalty(el)

    score = (
        (ppg * 2.0)
        + (form * 1.2)
        + (ict * 0.10)
        + (value * 3.0)
        + (fixture_ease * 1.0)
        + (min(minutes / 900.0, 1.0) * 1.0)
        - penalty
    )

    team_name = teams_by_id.get(team_id, {}).get("name", str(team_id))
    name = f"{el.get('first_name','')} {el.get('second_name','')}".strip() or str(el.get("web_name"))

    return {
        "id": int(el["id"]),
        "name": name,
        "team": team_name,
        "position": pos,
        "price_m": round(price, 1),
        "base_score": round(score, 3),
        "signals": {
            "points_per_game": ppg,
            "form": form,
            "ict_index": ict,
            "value_ppg_per_million": round(value, 3),
            "fixture_ease_next_horizon": round(fixture_ease, 3),
            "availability_penalty": penalty,
            "minutes_season": minutes,
        },
    }


def _recent_form_from_element_summary(es: dict[str, Any], last_matches: int) -> dict[str, Any]:
    """
    Extract recent trend signals from element-summary history.

    Works even if some expected fields are missing by falling back safely.
    """
    history: list[dict[str, Any]] = es.get("history", []) or []
    if not history:
        return {
            "matches_used": 0,
            "avg_minutes": 0.0,
            "points_per_90": 0.0,
            "xgi_per_90": 0.0,
            "blank_rate": None,
        }

    last = history[-max(1, min(last_matches, len(history))):]

    mins = sum(int(_to_float(h.get("minutes"))) for h in last)
    pts = sum(int(_to_float(h.get("total_points"))) for h in last)

    # Prefer expected_goal_involvements if present; otherwise fall back to expected_goals + expected_assists
    xgi_sum = 0.0
    blanks = 0
    for h in last:
        xgi = h.get("expected_goal_involvements", None)
        if xgi is not None:
            xgi_sum += _to_float(xgi)
        else:
            xgi_sum += _to_float(h.get("expected_goals")) + _to_float(h.get("expected_assists"))

        if int(_to_float(h.get("total_points"))) <= 2:
            blanks += 1

    matches_used = len(last)
    avg_minutes = mins / matches_used if matches_used else 0.0
    points_per_90 = (pts / mins) * 90.0 if mins > 0 else 0.0
    xgi_per_90 = (xgi_sum / mins) * 90.0 if mins > 0 else 0.0
    blank_rate = (blanks / matches_used) if matches_used else None

    return {
        "matches_used": matches_used,
        "avg_minutes": round(avg_minutes, 2),
        "points_per_90": round(points_per_90, 3),
        "xgi_per_90": round(xgi_per_90, 3),
        "blank_rate": None if blank_rate is None else round(blank_rate, 3),
    }


def _refine_score(base: dict[str, Any], recent: dict[str, Any]) -> dict[str, Any]:
    """
    Second-pass adjustment:
    - Reward: recent xGI/90, recent points/90, strong minutes trend
    - Penalise: low average minutes (rotation risk)
    """
    base_score = float(base.get("base_score", 0.0))
    avg_minutes = float(recent.get("avg_minutes", 0.0))
    p90 = float(recent.get("points_per_90", 0.0))
    xgi90 = float(recent.get("xgi_per_90", 0.0))

    minutes_factor = min(avg_minutes / 90.0, 1.0)
    rotation_pen = 0.0
    if avg_minutes > 0 and avg_minutes < 60:
        rotation_pen = 1.5
    elif avg_minutes >= 60 and avg_minutes < 75:
        rotation_pen = 0.6

    refined = (
        base_score
        + (xgi90 * 2.2)
        + (p90 * 0.9)
        + (minutes_factor * 1.2)
        - rotation_pen
    )

    out = dict(base)
    out["refined_score"] = round(refined, 3)
    out["recent_signals"] = recent
    out["adjustments"] = {
        "rotation_penalty": rotation_pen,
        "minutes_factor": round(minutes_factor, 3),
        "xgi90_weight": 2.2,
        "p90_weight": 0.9,
    }
    return out


def _require_bearer(request: Request) -> Response | None:
    if not BEARER_TOKEN:
        return None
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {BEARER_TOKEN}":
        return JSONResponse({"error": "unauthorised"}, status_code=401)
    return None


# --------------------
# MCP server + tools
# --------------------
server = Server("fpl-advisor")

TOOLS: list[Tool] = [
    Tool(
        name="fpl_find_players",
        description="Find FPL players by partial name match (first/second/web_name).",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Name search string"},
                "limit": {"type": "integer", "description": "Max results", "default": 10},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="fpl_player_summary",
        description="Player snapshot using element-summary history (recent minutes, points/90, xGI/90) + upcoming fixtures.",
        inputSchema={
            "type": "object",
            "properties": {
                "player_id": {"type": "integer", "description": "FPL element id"},
                "last_matches": {"type": "integer", "default": 5, "description": "How many recent matches to analyse"},
            },
            "required": ["player_id"],
        },
    ),
    Tool(
        name="fpl_best_players",
        description="Fast rank using bootstrap + fixtures only (good for quick shortlist).",
        inputSchema={
            "type": "object",
            "properties": {
                "position": {"type": "string", "description": "Optional: GKP/DEF/MID/FWD"},
                "max_price_m": {"type": "number", "description": "Optional price cap in £m"},
                "horizon_gws": {"type": "integer", "default": 5, "description": "Fixture horizon (gameweeks)"},
                "limit": {"type": "integer", "default": 25, "description": "Max results"},
                "min_minutes": {"type": "integer", "default": 0, "description": "Filter low-minutes players"},
                "include_unavailable": {"type": "boolean", "default": False, "description": "Include flagged/unavailable"},
            },
            "required": [],
        },
    ),
    Tool(
        name="fpl_best_players_refined",
        description="Two-pass rank: quick shortlist (bootstrap+fixtures) then refine with element-summary trends (minutes/points/90/xGI/90).",
        inputSchema={
            "type": "object",
            "properties": {
                "position": {"type": "string", "description": "Optional: GKP/DEF/MID/FWD"},
                "max_price_m": {"type": "number", "description": "Optional price cap in £m"},
                "horizon_gws": {"type": "integer", "default": 5, "description": "Fixture horizon (gameweeks)"},
                "limit": {"type": "integer", "default": 25, "description": "Max results"},
                "min_minutes": {"type": "integer", "default": 0, "description": "Filter low-minutes players"},
                "include_unavailable": {"type": "boolean", "default": False, "description": "Include flagged/unavailable"},
                "refine_pool": {"type": "integer", "default": 60, "description": "How many top candidates to enrich via element-summary"},
                "last_matches": {"type": "integer", "default": 5, "description": "Recent matches window for refinement"},
                "concurrency": {"type": "integer", "default": 8, "description": "Max concurrent element-summary fetches"},
            },
            "required": [],
        },
    ),
]


@server.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "fpl_find_players":
        q = str(arguments.get("query", "")).strip().lower()
        limit = int(arguments.get("limit", 10))

        data = await _bootstrap()
        teams_by_id = {int(t["id"]): t for t in data.get("teams", [])}

        hits = []
        for el in data.get("elements", []):
            full = f"{el.get('first_name','')} {el.get('second_name','')}".strip().lower()
            web = str(el.get("web_name", "")).lower()
            if q and (q in full or q in web):
                team_id = int(el["team"])
                hits.append(
                    {
                        "id": int(el["id"]),
                        "name": f"{el.get('first_name','')} {el.get('second_name','')}".strip()
                        or el.get("web_name"),
                        "web_name": el.get("web_name"),
                        "team": teams_by_id.get(team_id, {}).get("name", str(team_id)),
                        "position": POS_MAP.get(int(el["element_type"]), "MID"),
                        "price_m": round(_price_m(int(el["now_cost"])), 1),
                        "status": el.get("status"),
                    }
                )
            if len(hits) >= limit:
                break

        return [TextContent(type="text", text=json.dumps(hits, ensure_ascii=False))]

    if name == "fpl_player_summary":
        player_id = int(arguments.get("player_id"))
        last_matches = int(arguments.get("last_matches", 5))

        bs = await _bootstrap()
        teams_by_id = {int(t["id"]): t for t in bs.get("teams", [])}
        elements = bs.get("elements", [])

        el = next((x for x in elements if int(x.get("id")) == player_id), None)
        if el is None:
            return [TextContent(type="text", text=json.dumps({"error": f"Unknown player_id={player_id}"}))]

        es = await _element_summary(player_id)
        recent = _recent_form_from_element_summary(es, last_matches=last_matches)

        team_id = int(el["team"])
        payload = {
            "player": {
                "id": player_id,
                "name": f"{el.get('first_name','')} {el.get('second_name','')}".strip() or el.get("web_name"),
                "team": teams_by_id.get(team_id, {}).get("name", str(team_id)),
                "position": POS_MAP.get(int(el["element_type"]), "MID"),
                "price_m": round(_price_m(int(el["now_cost"])), 1),
                "status": el.get("status"),
                "chance_of_playing_next_round": el.get("chance_of_playing_next_round"),
            },
            "recent": recent,
            "upcoming_fixtures": es.get("fixtures", [])[:10],
            "history_count": len(es.get("history", []) or []),
        }
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    if name in ("fpl_best_players", "fpl_best_players_refined"):
        position = arguments.get("position")
        max_price_m = arguments.get("max_price_m")
        horizon_gws = int(arguments.get("horizon_gws", 5))
        limit = int(arguments.get("limit", 25))
        min_minutes = int(arguments.get("min_minutes", 0))
        include_unavailable = bool(arguments.get("include_unavailable", False))

        refine_pool = int(arguments.get("refine_pool", 60))
        last_matches = int(arguments.get("last_matches", 5))
        concurrency = int(arguments.get("concurrency", 8))

        bs = await _bootstrap()
        fx = await _fixtures()

        events = bs.get("events", [])
        current_event = _current_event_id(events)

        teams_by_id = {int(t["id"]): t for t in bs.get("teams", [])}
        elements = bs.get("elements", [])

        pos_filter: Position | None = None
        if isinstance(position, str) and position.strip():
            pos_filter = position.strip().upper()  # type: ignore[assignment]

        # First pass: fast score
        first_pass: list[dict[str, Any]] = []
        for el in elements:
            if pos_filter and POS_MAP.get(int(el["element_type"]), "MID") != pos_filter:
                continue
            if int(_to_float(el.get("minutes"))) < min_minutes:
                continue

            price = _price_m(int(el["now_cost"]))
            if max_price_m is not None and price > float(max_price_m):
                continue

            if not include_unavailable and str(el.get("status", "a")) != "a":
                continue

            first_pass.append(
                _score_player_first_pass(el, teams_by_id, fx, horizon_gws=horizon_gws, current_event=current_event)
            )

        first_pass.sort(key=lambda r: r["base_score"], reverse=True)

        # If caller asked for fast only, return immediately
        if name == "fpl_best_players":
            payload = {
                "method": "first_pass_composite_v1",
                "current_event": current_event,
                "results": first_pass[: max(1, limit)],
            }
            return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

        # Second pass: refine top pool with element-summary trends
        pool = first_pass[: max(1, min(refine_pool, len(first_pass)))]
        sem = asyncio.Semaphore(max(1, concurrency))

        async def enrich_one(item: dict[str, Any]) -> dict[str, Any]:
            pid = int(item["id"])
            async with sem:
                es = await _element_summary(pid)
            recent = _recent_form_from_element_summary(es, last_matches=last_matches)
            return _refine_score(item, recent)

        enriched = await asyncio.gather(*(enrich_one(it) for it in pool))

        # Rank by refined score, then trim to limit
        enriched.sort(key=lambda r: r.get("refined_score", r.get("base_score", 0.0)), reverse=True)
        results = enriched[: max(1, limit)]

        payload = {
            "method": "two_pass_refined_v1",
            "current_event": current_event,
            "params": {
                "horizon_gws": horizon_gws,
                "limit": limit,
                "min_minutes": min_minutes,
                "position": position,
                "max_price_m": max_price_m,
                "include_unavailable": include_unavailable,
                "refine_pool": refine_pool,
                "last_matches": last_matches,
                "concurrency": concurrency,
            },
            "results": results,
        }
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    raise ValueError(f"Unknown tool: {name}")


# --------------------
# SSE Transport + Starlette wiring
# --------------------
sse = SseServerTransport("/messages/")


async def handle_sse(request: Request) -> Response:
    auth_resp = _require_bearer(request)
    if auth_resp:
        return auth_resp

    async with sse.connect_sse(request.scope, request.receive, request._send) as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
    return Response(status_code=204)


async def health(_: Request) -> Response:
    return JSONResponse({"status": "ok"})


starlette_app = Starlette(
    debug=os.getenv("DEBUG", "0") == "1",
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/sse", handle_sse, methods=["GET"]),
        Mount("/messages/", app=sse.handle_post_message),
    ],
)


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(starlette_app, host=host, port=port, log_level=os.getenv("UVICORN_LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()
