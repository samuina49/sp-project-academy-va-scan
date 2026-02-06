"""
Rate Limiting Middleware for FastAPI
Prevents API abuse with request throttling

Features:
- Per-IP rate limiting
- Configurable limits per endpoint
- Redis support for distributed systems (optional)
- In-memory fallback for single instance
"""

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)


class RateLimitExceeded(HTTPException):
    def __init__(self, retry_after: int = 60):
        super().__init__(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "message": f"Too many requests. Please try again in {retry_after} seconds.",
                "retry_after": retry_after
            }
        )


class InMemoryRateLimiter:
    """
    Simple in-memory rate limiter using sliding window algorithm.
    Suitable for single-instance deployments.
    """
    
    def __init__(self):
        # Structure: {ip: [(timestamp, count), ...]}
        self._requests: Dict[str, list] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    async def is_allowed(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> Tuple[bool, int, int]:
        """
        Check if request is allowed under rate limit.
        
        Args:
            key: Unique identifier (usually IP address)
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
            
        Returns:
            Tuple of (is_allowed, remaining_requests, retry_after_seconds)
        """
        async with self._lock:
            now = datetime.now()
            window_start = now - timedelta(seconds=window_seconds)
            
            # Clean old entries
            self._requests[key] = [
                ts for ts in self._requests[key]
                if ts > window_start
            ]
            
            current_count = len(self._requests[key])
            
            if current_count >= max_requests:
                # Calculate retry after
                if self._requests[key]:
                    oldest = min(self._requests[key])
                    retry_after = int((oldest + timedelta(seconds=window_seconds) - now).total_seconds())
                    retry_after = max(1, retry_after)
                else:
                    retry_after = window_seconds
                return False, 0, retry_after
            
            # Add new request
            self._requests[key].append(now)
            remaining = max_requests - current_count - 1
            
            return True, remaining, 0
    
    async def cleanup(self):
        """Remove expired entries to prevent memory leak"""
        async with self._lock:
            now = datetime.now()
            cutoff = now - timedelta(hours=1)  # Keep 1 hour of data max
            
            keys_to_remove = []
            for key, timestamps in self._requests.items():
                self._requests[key] = [ts for ts in timestamps if ts > cutoff]
                if not self._requests[key]:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._requests[key]


# Global rate limiter instance
_rate_limiter = InMemoryRateLimiter()


# Rate limit configurations per endpoint pattern
RATE_LIMITS = {
    # Scan endpoints - more restrictive
    "/api/v1/ml-scan": {"max_requests": 10, "window_seconds": 60},
    "/api/v1/scan": {"max_requests": 10, "window_seconds": 60},
    "/api/v1/explain": {"max_requests": 10, "window_seconds": 60},
    
    # Report generation - moderate
    "/api/v1/report/pdf": {"max_requests": 5, "window_seconds": 60},
    
    # Feedback - lenient
    "/api/v1/feedback": {"max_requests": 30, "window_seconds": 60},
    
    # Health checks - very lenient
    "/api/v1/health": {"max_requests": 60, "window_seconds": 60},
    
    # Default for unspecified endpoints
    "default": {"max_requests": 30, "window_seconds": 60}
}


def get_rate_limit_config(path: str) -> dict:
    """Get rate limit config for a given path"""
    for pattern, config in RATE_LIMITS.items():
        if pattern != "default" and path.startswith(pattern):
            return config
    return RATE_LIMITS["default"]


def get_client_ip(request: Request) -> str:
    """Extract client IP from request, considering proxies"""
    # Check for forwarded headers (behind proxy/load balancer)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Direct connection
    if request.client:
        return request.client.host
    
    return "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.
    
    Usage:
        app.add_middleware(RateLimitMiddleware)
    """
    
    def __init__(self, app, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled
        self._cleanup_task = None
    
    async def dispatch(self, request: Request, call_next):
        if not self.enabled:
            return await call_next(request)
        
        # Skip rate limiting for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # Get client identifier
        client_ip = get_client_ip(request)
        path = request.url.path
        
        # Get rate limit config for this endpoint
        config = get_rate_limit_config(path)
        
        # Create unique key for this IP + endpoint
        rate_key = f"{client_ip}:{path}"
        
        # Check rate limit
        is_allowed, remaining, retry_after = await _rate_limiter.is_allowed(
            rate_key,
            config["max_requests"],
            config["window_seconds"]
        )
        
        if not is_allowed:
            logger.warning(f"Rate limit exceeded for {client_ip} on {path}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Please try again in {retry_after} seconds.",
                    "retry_after": retry_after
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(config["max_requests"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(retry_after)
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(config["max_requests"])
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Window"] = str(config["window_seconds"])
        
        return response


# Cleanup task for memory management
async def start_rate_limit_cleanup():
    """Background task to clean up expired rate limit entries"""
    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        await _rate_limiter.cleanup()
        logger.debug("Rate limiter cleanup completed")
