import sys
sys.path.insert(0, ".")
from sprint_planning_env.server.app import app

for route in app.routes:
    methods = getattr(route, 'methods', None)
    path = getattr(route, 'path', None)
    print(f"{methods} {path}")
