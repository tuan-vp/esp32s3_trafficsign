idf_component_register(
    SRCS main.cpp model.cc
    INCLUDE_DIRS .
    REQUIRES esp-tflite-micro
    PRIV_REQUIRES esp_timer driver led_strip esp_wifi esp_http_server nvs_flash
)