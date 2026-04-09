### How to run the app?
docker compose up -d --build  
<br>

### How to run on GPU or CPU?
To choose between GPU and CPU, go to the `docker-compose.yaml` file and comment or uncomment the appropriate section of the code.  
<br>

### How to change the Whisper model?
To change the Whisper model, update the appropriate value in the `config.env` file.  
<br>

### How to change the language?
For better model accuracy, a single language is selected. If you want to change it, you can do so in the `config.env` file.  
<br>
