# ==============================================================================
# AI-Based Vulnerability Scanner - Frontend Dockerfile
# ==============================================================================
# Multi-stage build for Next.js application
# Stage 1: Dependencies
# Stage 2: Builder
# Stage 3: Production runner
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Dependencies
# ------------------------------------------------------------------------------
FROM node:18-alpine AS deps

WORKDIR /app

# Install dependencies based on the preferred package manager
COPY package.json package-lock.json* ./
RUN npm ci --only=production

# ------------------------------------------------------------------------------
# Stage 2: Builder
# ------------------------------------------------------------------------------
FROM node:18-alpine AS builder

WORKDIR /app

# Copy dependencies
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Environment variables for build
ENV NEXT_TELEMETRY_DISABLED=1
ENV NODE_ENV=production

# Build the application
RUN npm run build

# ------------------------------------------------------------------------------
# Stage 3: Production Runner
# ------------------------------------------------------------------------------
FROM node:18-alpine AS runner

WORKDIR /app

# Labels
LABEL maintainer="AI Vulnerability Scanner Team"
LABEL version="1.0.0"
LABEL description="AI-Based Vulnerability Scanner Frontend"

# Security: Run as non-root user
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

# Environment variables
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000

# Copy necessary files from builder
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

# Set ownership
RUN chown -R nextjs:nodejs /app

# Expose port
EXPOSE 3000

# Switch to non-root user
USER nextjs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/ || exit 1

# Run the application
CMD ["node", "server.js"]

# ==============================================================================
# Note: Requires output: 'standalone' in next.config.js
# 
# Add to next.config.js:
#   module.exports = {
#     output: 'standalone',
#   }
#
# Build Commands:
#   docker build -t vuln-scanner-frontend:latest .
#
# Run:
#   docker run -p 3000:3000 vuln-scanner-frontend:latest
# ==============================================================================
