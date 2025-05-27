from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import os
import hashlib
import hmac
from datetime import datetime, timedelta
import secrets

# Security scheme
security = HTTPBearer()

class AuthManager:
    def __init__(self):
        # Load API keys from environment
        self.master_api_key = os.getenv("MASTER_API_KEY", "master-dev-key")
        self.api_keys = self._load_api_keys()
        
    def _load_api_keys(self) -> set:
        """Load valid API keys from environment or database"""
        keys = set()
        
        # Add master key if provided
        if self.master_api_key:
            keys.add(self.master_api_key)
        
        # Add additional keys from environment (comma-separated)
        additional_keys = os.getenv("ADDITIONAL_API_KEYS", "")
        if additional_keys:
            keys.update(key.strip() for key in additional_keys.split(",") if key.strip())
        
        # Always add development keys for testing
        dev_keys = [
            "dev-api-key-123",
            "test-api-key-456", 
            "demo-key-789",
            "qa-development-key"
        ]
        keys.update(dev_keys)
        
        print(f"✅ Loaded {len(keys)} API keys for authentication")
        return keys
    
    def verify_api_key(self, api_key: str) -> bool:
        """Verify if the provided API key is valid"""
        is_valid = api_key in self.api_keys
        print(f"🔐 Verifying API key {api_key[:8]}...: {'✅ Valid' if is_valid else '❌ Invalid'}")
        return is_valid
    
    def generate_api_key(self) -> str:
        """Generate a new API key"""
        return f"qa-{secrets.token_urlsafe(32)}"
    
    def add_api_key(self, api_key: str) -> bool:
        """Add a new API key to the valid keys set"""
        self.api_keys.add(api_key)
        return True
    
    def remove_api_key(self, api_key: str) -> bool:
        """Remove an API key from the valid keys set"""
        if api_key in self.api_keys and api_key != self.master_api_key:
            self.api_keys.remove(api_key)
            return True
        return False
    
    def list_api_keys(self) -> list:
        """List all API keys (masked for security)"""
        return [f"{key[:8]}...{key[-4:]}" if len(key) > 12 else f"{key[:4]}..." for key in self.api_keys]

# Global auth manager instance
auth_manager = AuthManager()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Verify API key from Authorization header
    Expected format: Authorization: Bearer <api_key>
    """
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    api_key = credentials.credentials
    
    if not auth_manager.verify_api_key(api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return api_key

def verify_api_key_header(api_key: Optional[str] = None) -> str:
    """
    Alternative verification method using custom header
    Expected format: X-API-Key: <api_key>
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key missing in X-API-Key header",
        )
    
    if not auth_manager.verify_api_key(api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )
    
    return api_key

# Rate limiting per API key
class RateLimiter:
    def __init__(self):
        self.requests = {}  # api_key -> list of timestamps
        self.max_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
        self.time_window = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))  # 1 hour
    
    def is_allowed(self, api_key: str) -> bool:
        """Check if the API key is within rate limits"""
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self.time_window)
        
        # Clean old requests
        if api_key in self.requests:
            self.requests[api_key] = [
                req_time for req_time in self.requests[api_key] 
                if req_time > cutoff
            ]
        else:
            self.requests[api_key] = []
        
        # Check if under limit
        if len(self.requests[api_key]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[api_key].append(now)
        return True

rate_limiter = RateLimiter()

def check_rate_limit(api_key: str = Depends(verify_api_key)) -> str:
    """Check rate limiting for the API key"""
    if not rate_limiter.is_allowed(api_key):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
        )
    return api_key

print("✅ Authentication module loaded:")
print(f"- API keys configured: {len(auth_manager.api_keys)}")
print(f"- Rate limiting: {rate_limiter.max_requests} requests per {rate_limiter.time_window} seconds")
