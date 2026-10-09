# Setup
## Installing ESP-IDF
- Set target DevKit, in this project is ESP32-S3
```bash
idf.py set-target esp32s3
```
- Clone this repo:
```
git clone https://github.com/vietdai-bk/esp32_c_infer_script
```
- Change ```SSID``` and ```password``` on ```wifi_config.h```
- Copy ```model.cc``` file on ```main``` folder and change ```model.h``` based on ```<model> array```  
Example: You has ```model.cc``` file with ***struct***:
```
unsigned char traffic_sign_int8_tflite[] = {<hex code>};
unsigned int traffic_sign_int8_tflite_len = <len of C array>;
```
You may write ```model.h``` file:
```
#pragma once
extern const unsigned char traffic_sign_int8_tflite[];
extern const unsigned int traffic_sign_int8_tflite_len;
```
- Add this in VS Code
- Installing dependency and Build 
```
idf.py build
```
 - Flash Code on ESP32:
```
idf.py -p <PORT> flash
```
Example: ```idf.py -p COM11 flash```  
- Monitoring:
```
idf.py monitor
```
Trick: You can skip ```idf.py build``` and run this code:
```
idf.py set-target esp32s3
idf.py -p COM11 flash monitor
```
