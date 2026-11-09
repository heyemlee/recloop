# Web image for the kind cluster (M5, deploy/helm/recloop): the store as a standalone Next.js
# production build. Build context: web/ (deploy/web.Dockerfile.dockerignore keeps node_modules
# and build output out). The rewrites' targets are fixed at build time: /api/v1/* goes to the api
# Service. The shopping assistant (agent/server.py) is not deployed on the cluster (design
# §23.2's service list; its encoder needs torch): RECLOOP_AGENT_URL is empty, so there is no
# /api/v1/chat rewrite and the chat page says where the assistant runs. Node 26.7.0, the version
# the app is developed with.
FROM node:26.7.0-trixie-slim@sha256:5758d367d7b4f48b73a9bb3530e687e47efb289f3b43f9c0450a25225ae0db5d AS build
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
ARG RECLOOP_API_URL=http://api:18080
ARG RECLOOP_AGENT_URL=
RUN NEXT_OUTPUT=standalone RECLOOP_API_URL=$RECLOOP_API_URL RECLOOP_AGENT_URL=$RECLOOP_AGENT_URL \
    npx next build

FROM node:26.7.0-trixie-slim@sha256:5758d367d7b4f48b73a9bb3530e687e47efb289f3b43f9c0450a25225ae0db5d
WORKDIR /app
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 PORT=3000 HOSTNAME=0.0.0.0
COPY --from=build --chown=1000:1000 /app/.next/standalone ./
COPY --from=build --chown=1000:1000 /app/.next/static ./.next/static
# The image's "node" user.
USER 1000:1000
EXPOSE 3000
CMD ["node", "server.js"]
