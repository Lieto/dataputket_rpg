FROM node:alpine 

WORKDIR /app 

ENV PATH /app/node_modules/.bin:$PATH   

COPY ./package.json /.
RUN npm install 

COPY . .

RUN npm install axios 

EXPOSE 3000 

CMD ["npm", "run", "dev"]
