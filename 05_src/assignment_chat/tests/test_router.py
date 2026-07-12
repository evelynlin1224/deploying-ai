from services.router import Route, route_message


def test_weather_command_routes_to_api():
    assert route_message("/weather Toronto") == Route.API_WEATHER


def test_kb_command_routes_to_semantic():
    assert route_message("/kb How does Chroma work?") == Route.SEMANTIC_KB


def test_plan_routes_to_tool():
    assert route_message("Make a 5 day plan for this assignment") == Route.PLANNER_TOOL
