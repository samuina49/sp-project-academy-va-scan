"""
JWT Authentication System for Multi-User Support
Provides secure token-based authentication

Features:
- JWT token generation and validation
- Password hashing with bcrypt
- Role-based access control (RBAC)
- Token refresh mechanism
- API key support for CI/CD integration
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, List
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from pydantic import BaseModel, EmailStr
from jose import JWTError, jwt
import bcrypt
import secrets
import logging

logger = logging.getLogger(__name__)

# Configuration (should be in environment variables in production)
SECRET_KEY = secrets.token_hex(32)  # Generate secure key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7
API_KEY_HEADER = "X-API-Key"

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


# ==================== Models ====================

class UserRole:
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"
    API = "api"  # For CI/CD integration


class User(BaseModel):
    id: str
    email: EmailStr
    username: str
    role: str = UserRole.USER
    is_active: bool = True
    created_at: datetime = None
    
    class Config:
        from_attributes = True


class UserInDB(User):
    hashed_password: str


class Token(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    user_id: str
    email: str
    role: str
    exp: datetime


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RegisterRequest(BaseModel):
    email: EmailStr
    username: str
    password: str


class APIKeyCreate(BaseModel):
    name: str
    expires_days: int = 365


class APIKey(BaseModel):
    id: str
    name: str
    key_prefix: str  # First 8 chars for identification
    user_id: str
    created_at: datetime
    expires_at: datetime
    is_active: bool = True


# ==================== In-Memory Storage (Replace with DB in production) ====================

# Simulated user database
_users_db: Dict[str, UserInDB] = {}
_api_keys_db: Dict[str, dict] = {}


def init_default_users():
    """Initialize default admin user for demo"""
    if "admin@example.com" not in _users_db:
        _users_db["admin@example.com"] = UserInDB(
            id="1",
            email="admin@example.com",
            username="admin",
            role=UserRole.ADMIN,
            hashed_password=get_password_hash("admin123"),
            created_at=datetime.utcnow()
        )
        logger.info("Default admin user created")


# ==================== Password Functions ====================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return bcrypt.checkpw(
        plain_password.encode('utf-8'), 
        hashed_password.encode('utf-8')
    )


def get_password_hash(password: str) -> str:
    """Hash a password for storage"""
    return bcrypt.hashpw(
        password.encode('utf-8'), 
        bcrypt.gensalt()
    ).decode('utf-8')


# ==================== Token Functions ====================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """Create a JWT refresh token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[TokenData]:
    """Decode and validate a JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        email = payload.get("email")
        role = payload.get("role", UserRole.USER)
        exp = datetime.fromtimestamp(payload.get("exp"))
        
        if user_id is None:
            return None
        
        return TokenData(user_id=user_id, email=email, role=role, exp=exp)
    except JWTError as e:
        logger.warning(f"Token decode error: {e}")
        return None


# ==================== API Key Functions ====================

def generate_api_key() -> str:
    """Generate a secure API key"""
    return f"vsc_{secrets.token_urlsafe(32)}"


def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage"""
    return pwd_context.hash(api_key)


def verify_api_key(api_key: str) -> Optional[dict]:
    """Verify an API key and return associated data"""
    for key_id, key_data in _api_keys_db.items():
        if pwd_context.verify(api_key, key_data["hashed_key"]):
            if key_data.get("is_active", True):
                if key_data["expires_at"] > datetime.utcnow():
                    return key_data
    return None


# ==================== User Functions ====================

def get_user(email: str) -> Optional[UserInDB]:
    """Get user by email"""
    return _users_db.get(email)


def create_user(email: str, username: str, password: str, role: str = UserRole.USER) -> User:
    """Create a new user"""
    if email in _users_db:
        raise ValueError("User already exists")
    
    user = UserInDB(
        id=str(len(_users_db) + 1),
        email=email,
        username=username,
        role=role,
        hashed_password=get_password_hash(password),
        created_at=datetime.utcnow()
    )
    _users_db[email] = user
    
    return User(**user.dict(exclude={"hashed_password"}))


def authenticate_user(email: str, password: str) -> Optional[UserInDB]:
    """Authenticate user with email and password"""
    user = get_user(email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


# ==================== Dependency Injection ====================

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
    api_key: str = Security(api_key_header)
) -> User:
    """
    Get current authenticated user from JWT token or API key.
    
    Usage:
        @router.get("/protected")
        async def protected_route(current_user: User = Depends(get_current_user)):
            return {"user": current_user}
    """
    # Try JWT token first
    if credentials:
        token_data = decode_token(credentials.credentials)
        if token_data:
            user = get_user(token_data.email)
            if user and user.is_active:
                return User(**user.dict(exclude={"hashed_password"}))
    
    # Try API key
    if api_key:
        key_data = verify_api_key(api_key)
        if key_data:
            user = get_user(key_data["user_email"])
            if user:
                return User(**user.dict(exclude={"hashed_password"}))
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Ensure user is active"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


def require_role(allowed_roles: List[str]):
    """
    Dependency for role-based access control.
    
    Usage:
        @router.get("/admin", dependencies=[Depends(require_role([UserRole.ADMIN]))])
        async def admin_only():
            return {"message": "Admin access"}
    """
    async def role_checker(current_user: User = Depends(get_current_user)):
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return role_checker


# ==================== Optional Auth (for public endpoints with optional auth) ====================

async def get_optional_user(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
    api_key: str = Security(api_key_header)
) -> Optional[User]:
    """
    Get current user if authenticated, None otherwise.
    Use for endpoints that work for both authenticated and anonymous users.
    """
    try:
        return await get_current_user(credentials, api_key)
    except HTTPException:
        return None


# Initialize default users on module load
init_default_users()
