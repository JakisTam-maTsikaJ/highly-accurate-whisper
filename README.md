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

### Can I transcribe multi-channel audio?
Yes, you can. Information about the channels will be included in the output. However, keep in mind that for multi-channel audio, channels are selected based on signal energy. If two people speak at the same time, the channel assignment may be inaccurate.
<br>

### More details
More details, such as the output format and how to send a request, can be found in Swagger.
