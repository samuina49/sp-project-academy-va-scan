"""
Authentication API Endpoints
Provides login, register, token refresh, and API key management
"""

from fastapi import APIRouter, HTTPException, Depends, status
from datetime import datetime, timedelta
from typing import List

from app.core.auth import (
    User, Token, LoginRequest, RegisterRequest, APIKeyCreate, APIKey,
    authenticate_user, create_user, create_access_token, create_refresh_token,
    decode_token, get_current_user, get_current_active_user, require_role,
    generate_api_key, hash_api_key, UserRole, ACCESS_TOKEN_EXPIRE_MINUTES,
    _api_keys_db
)

router = APIRouter()


@router.post("/auth/login", response_model=Token, tags=["Authentication"])
async def login(request: LoginRequest):
    """
    **User Login**
    
    Authenticate with email and password to receive JWT tokens.
    
    Returns:
    - access_token: Short-lived token for API access
    - refresh_token: Long-lived token for getting new access tokens
    """
    user = authenticate_user(request.email, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": user.id, "email": user.email, "role": user.role}
    )
    refresh_token = create_refresh_token(
        data={"sub": user.id, "email": user.email, "role": user.role}
    )
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/auth/register", response_model=User, tags=["Authentication"])
async def register(request: RegisterRequest):
    """
    **User Registration**
    
    Create a new user account.
    """
    try:
        user = create_user(
            email=request.email,
            username=request.username,
            password=request.password
        )
        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/auth/refresh", response_model=Token, tags=["Authentication"])
async def refresh_token(refresh_token: str):
    """
    **Refresh Access Token**
    
    Use a valid refresh token to get a new access token.
    """
    token_data = decode_token(refresh_token)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    access_token = create_access_token(
        data={"sub": token_data.user_id, "email": token_data.email, "role": token_data.role}
    )
    
    return Token(
        access_token=access_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.get("/auth/me", response_model=User, tags=["Authentication"])
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """
    **Get Current User**
    
    Returns the authenticated user's information.
    """
    return current_user


@router.post("/auth/api-keys", response_model=dict, tags=["Authentication"])
async def create_api_key(
    request: APIKeyCreate,
    current_user: User = Depends(get_current_active_user)
):
    """
    **Create API Key**
    
    Generate an API key for CI/CD integration.
    The key is only shown once - store it securely!
    """
    api_key = generate_api_key()
    key_id = f"key_{len(_api_keys_db) + 1}"
    
    _api_keys_db[key_id] = {
        "id": key_id,
        "name": request.name,
        "hashed_key": hash_api_key(api_key),
        "user_id": current_user.id,
        "user_email": current_user.email,
        "created_at": datetime.utcnow(),
        "expires_at": datetime.utcnow() + timedelta(days=request.expires_days),
        "is_active": True
    }
    
    return {
        "id": key_id,
        "name": request.name,
        "api_key": api_key,  # Only shown once!
        "key_prefix": api_key[:12],
        "expires_at": _api_keys_db[key_id]["expires_at"].isoformat(),
        "message": "Store this API key securely - it won't be shown again!"
    }


@router.get("/auth/api-keys", response_model=List[APIKey], tags=["Authentication"])
async def list_api_keys(current_user: User = Depends(get_current_active_user)):
    """
    **List API Keys**
    
    Get all API keys for the current user.
    """
    user_keys = []
    for key_id, key_data in _api_keys_db.items():
        if key_data["user_id"] == current_user.id:
            user_keys.append(APIKey(
                id=key_data["id"],
                name=key_data["name"],
                key_prefix=key_data.get("key_prefix", "****"),
                user_id=key_data["user_id"],
                created_at=key_data["created_at"],
                expires_at=key_data["expires_at"],
                is_active=key_data["is_active"]
            ))
    return user_keys


@router.delete("/auth/api-keys/{key_id}", tags=["Authentication"])
async def revoke_api_key(
    key_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    **Revoke API Key**
    
    Deactivate an API key.
    """
    if key_id not in _api_keys_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
    
    key_data = _api_keys_db[key_id]
    if key_data["user_id"] != current_user.id and current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot revoke another user's API key"
        )
    
    _api_keys_db[key_id]["is_active"] = False
    
    return {"message": "API key revoked successfully"}


# Admin-only endpoints
@router.get(
    "/auth/users",
    response_model=List[User],
    tags=["Authentication"],
    dependencies=[Depends(require_role([UserRole.ADMIN]))]
)
async def list_users():
    """
    **List All Users** (Admin Only)
    
    Get all registered users.
    """
    from app.core.auth import _users_db
    return [User(**u.dict(exclude={"hashed_password"})) for u in _users_db.values()]
