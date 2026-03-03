from datetime import datetime, timedelta
from fastapi import Request
from fastapi.responses import JSONResponse

IP_THROTTLER = {}
MIN_WAITING_TIME = timedelta(seconds=3)
MAX_REQ_PER_MIN = 3
MAX_REQ_PER_DAY = 10

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
        
        if request.method == "OPTIONS":
            await self.app(scope, receive, send)
            return

        if ip in IP_THROTTLER:
            last_request = IP_THROTTLER[ip][-1]
            delay = now - last_request
            requests_in_last_minute = len([date for date in IP_THROTTLER[ip] if (now - date) < timedelta(minutes=1)])
            requests_in_last_day = len([date for date in IP_THROTTLER[ip] if (now - date) < timedelta(days=1)])
            if delay < MIN_WAITING_TIME:
                print(f"\033[91mTHROTTLED {ip} — {str(delay)} elapsed since last request\033[0m")
                response = JSONResponse(
                    status_code=429,
                    content={"error": "Ddos protection reached"}
                )
                await response(scope, receive, send)
            elif requests_in_last_minute > MAX_REQ_PER_MIN:
                print(f"\033[91mTHROTTLED {ip} — {requests_in_last_minute} requests in last minute\033[0m")
                response = JSONResponse(
                    status_code=429,
                    content={"error": "Max requests per minute reached"}
                )
                await response(scope, receive, send)
            elif requests_in_last_day > MAX_REQ_PER_DAY:
                print(f"\033[91mTHROTTLED {ip} — {requests_in_last_day} requests in last day\033[0m")
                response = JSONResponse(
                    status_code=429,
                    content={"error": "Max requests per day reached"}
                )
                await response(scope, receive, send)
            IP_THROTTLER[ip].append(now)
        else:
            IP_THROTTLER[ip] = [now]
            
        await self.app(scope, receive, send)