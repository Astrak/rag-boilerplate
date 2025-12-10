from datetime import datetime, timedelta
from fastapi import Request
from fastapi.responses import JSONResponse

IP_THROTTLER = {}
COOLDOWN = timedelta(seconds=3)

class IPThrottleMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        ip = request.client.host
        now = datetime.utcnow()

        if ip in IP_THROTTLER and now - IP_THROTTLER[ip] < COOLDOWN:
            response = JSONResponse(
                status_code=429,
                content={"error": "Rate limit reached"}
            )
            await response(scope, receive, send)
            return

        IP_THROTTLER[ip] = now
        await self.app(scope, receive, send)