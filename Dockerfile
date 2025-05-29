FROM node:18-alpine

WORKDIR /app 

# Copy package.json and install dependencies
COPY package.json package-lock.json ./
RUN npm ci 

# Copy the rest of the application
COPY . .

# Build the application
RUN npm run build 

# Install serve to run the application
RUN npm install -g serve 

# Expose the port the app runs on 
EXPOSE 5173

# COmmand to run the application
CMD ["serve", "-s", "dist", "-l", "5173"]