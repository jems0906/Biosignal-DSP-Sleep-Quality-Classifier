FROM node:20-slim AS build
WORKDIR /app
ARG VITE_API_URL=http://127.0.0.1:8000
ENV VITE_API_URL=$VITE_API_URL
COPY package.json .npmrc ./
COPY package-lock.json ./
RUN npm ci --include=optional
COPY . .
RUN npm run build
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
