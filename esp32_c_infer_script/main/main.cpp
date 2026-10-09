#include <stdio.h>
#include <string.h>
#include "model_config.h"
#include "wifi_config.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_http_server.h"
#include "esp_heap_caps.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define IMG_BYTES   (IMG_W * IMG_H * IMG_CHANNELS)

static const char* TAG = "EDGE_AI";
static uint8_t* tensor_arena                   = nullptr;
static tflite::MicroMutableOpResolver<RESOLVER_SIZE> resolver;
static tflite::MicroInterpreter* g_interpreter = nullptr;
static EventGroupHandle_t s_wifi_event_group;

#define WIFI_CONNECTED_BIT  BIT0
#define WIFI_FAIL_BIT       BIT1
static int s_retry_num = 0;

static void wifi_event_handler(void* arg, esp_event_base_t base,
                                int32_t id, void* data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();

    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        if (s_retry_num < WIFI_MAX_RETRY) {
            esp_wifi_connect();
            s_retry_num++;
            ESP_LOGW(TAG, "Wi-Fi mất kết nối, thử lại %d/%d...",
                     s_retry_num, WIFI_MAX_RETRY);
        } else {
            xEventGroupSetBits(s_wifi_event_group, WIFI_FAIL_BIT);
            ESP_LOGE(TAG, "Kết nối Wi-Fi thất bại sau %d lần thử.", WIFI_MAX_RETRY);
        }

    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* ev = (ip_event_got_ip_t*)data;
        s_retry_num = 0;
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
        ESP_LOGI(TAG, "╔══════════════════════════════════╗");
        ESP_LOGI(TAG, "  ESP32 IP : " IPSTR, IP2STR(&ev->ip_info.ip));
        ESP_LOGI(TAG, "  Endpoint : http://" IPSTR "/infer", IP2STR(&ev->ip_info.ip));
        ESP_LOGI(TAG, "╚══════════════════════════════════╝");
    }
}

static bool wifi_init_sta()
{
    s_wifi_event_group = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, NULL));

    wifi_config_t wifi_cfg = {};
    strncpy((char*)wifi_cfg.sta.ssid,     WIFI_SSID, sizeof(wifi_cfg.sta.ssid) - 1);
    strncpy((char*)wifi_cfg.sta.password, WIFI_PASS, sizeof(wifi_cfg.sta.password) - 1);
    wifi_cfg.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_cfg));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Đang kết nối Wi-Fi SSID: %s ...", WIFI_SSID);

    EventBits_t bits = xEventGroupWaitBits(
        s_wifi_event_group,
        WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
        pdFALSE, pdFALSE,
        pdMS_TO_TICKS(15000)
    );

    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "Wi-Fi OK: %s", WIFI_SSID);
        return true;
    }
    ESP_LOGE(TAG, "Wi-Fi FAIL");
    return false;
}

static inline int get_output_score(TfLiteTensor* output, int idx)
{
#ifdef MODEL_DTYPE_FLOAT32
    return (int)(output->data.f[idx] * 255.0f);
#else
    return (int)output->data.uint8[idx];
#endif
}

static esp_err_t infer_post_handler(httpd_req_t* req)
{
    int64_t t_start = esp_timer_get_time();

    if (req->content_len != IMG_BYTES) {
        ESP_LOGW(TAG, "Content-Length sai: got=%d expect=%d",
                 req->content_len, IMG_BYTES);
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST,
                            "Body size mismatch — check IMG_W/H/CHANNELS in model_config.h");
        return ESP_FAIL;
    }

    TfLiteTensor* input = g_interpreter->input(0);

#ifdef MODEL_DTYPE_FLOAT32
    static uint8_t raw_buf[IMG_BYTES];
    uint8_t* recv_ptr = raw_buf;
#else
    uint8_t* recv_ptr = input->data.uint8;
#endif

    int received = 0;
    while (received < IMG_BYTES) {
        int n = httpd_req_recv(req,
                               (char*)(recv_ptr + received),
                               IMG_BYTES - received);
        if (n <= 0) {
            ESP_LOGE(TAG, "httpd_req_recv lỗi: %d", n);
            httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Receive error");
            return ESP_FAIL;
        }
        received += n;
    }

#ifdef MODEL_DTYPE_FLOAT32
    for (int i = 0; i < IMG_BYTES; i++)
        input->data.f[i] = raw_buf[i] / 255.0f;
#endif

    int64_t t_recv_done   = esp_timer_get_time();
    int64_t t_infer_start = esp_timer_get_time();

    if (g_interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke() thất bại");
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Inference failed");
        return ESP_FAIL;
    }

    int64_t t_infer_end = esp_timer_get_time();

    TfLiteTensor* output = g_interpreter->output(0);
    int best_class = 0, best_score = 0;
    for (int i = 0; i < NUM_CLASSES; i++) {
        int s = get_output_score(output, i);
        if (s > best_score) { best_score = s; best_class = i; }
    }

    int64_t t_end = esp_timer_get_time();

    float recv_ms  = (t_recv_done  - t_start)        / 1000.0f;
    float infer_ms = (t_infer_end  - t_infer_start)   / 1000.0f;
    float total_ms = (t_end        - t_start)          / 1000.0f;

    ESP_LOGI(TAG, "▶ class=%d score=%d | recv=%.1fms infer=%.1fms total=%.1fms",
             best_class, best_score, recv_ms, infer_ms, total_ms);

    char scores_buf[NUM_CLASSES * 6 + 4];
    scores_buf[0] = '['; scores_buf[1] = '\0';
    for (int i = 0; i < NUM_CLASSES; i++) {
        char tmp[8];
        snprintf(tmp, sizeof(tmp), "%d", get_output_score(output, i));
        strcat(scores_buf, tmp);
        if (i < NUM_CLASSES - 1) strcat(scores_buf, ",");
    }
    strcat(scores_buf, "]");

    char resp_buf[256];
    snprintf(resp_buf, sizeof(resp_buf),
             "{"
             "\"class\":%d,"
             "\"score\":%d,"
             "\"scores\":%s,"
             "\"recv_ms\":%.2f,"
             "\"infer_ms\":%.2f,"
             "\"total_ms\":%.2f"
             "}",
             best_class, best_score, scores_buf,
             recv_ms, infer_ms, total_ms);

    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, resp_buf);
    return ESP_OK;
}

static esp_err_t ping_get_handler(httpd_req_t* req)
{
    char buf[128];
    snprintf(buf, sizeof(buf),
             "{\"status\":\"ok\",\"img\":[%d,%d,%d],\"classes\":%d}",
             IMG_W, IMG_H, IMG_CHANNELS, NUM_CLASSES);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, buf);
    return ESP_OK;
}

static httpd_handle_t start_webserver()
{
    httpd_config_t config    = HTTPD_DEFAULT_CONFIG();
    config.server_port       = 80;
    config.recv_wait_timeout = 15;
    config.send_wait_timeout = 15;
    config.max_open_sockets  = 4;

    httpd_handle_t server = NULL;
    if (httpd_start(&server, &config) != ESP_OK) {
        ESP_LOGE(TAG, "httpd_start thất bại");
        return NULL;
    }

    // static const httpd_uri_t infer_uri = {
    //     .uri     = "/infer",
    //     .method  = HTTP_POST,
    //     .handler = infer_post_handler,
    // };
    // static const httpd_uri_t ping_uri = {
    //     .uri     = "/ping",
    //     .method  = HTTP_GET,
    //     .handler = ping_get_handler,
    // };

    static const httpd_uri_t infer_uri = {
        .uri      = "/infer",
        .method   = HTTP_POST,
        .handler  = infer_post_handler,
        .user_ctx = nullptr,
    };

    static const httpd_uri_t ping_uri = {
        .uri      = "/ping",
        .method   = HTTP_GET,
        .handler  = ping_get_handler,
        .user_ctx = nullptr,
    };

    httpd_register_uri_handler(server, &infer_uri);
    httpd_register_uri_handler(server, &ping_uri);

    ESP_LOGI(TAG, "HTTP server lắng nghe cổng 80");
    return server;
}

extern "C" void app_main()
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
        ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    const int arena_size = TENSOR_ARENA_KB * 1024;
    tensor_arena = (uint8_t*)heap_caps_malloc(arena_size, MALLOC_CAP_SPIRAM);
    if (tensor_arena == nullptr) {
        ESP_LOGE(TAG, "Không cấp phát được %dKB trên PSRAM!", TENSOR_ARENA_KB);
        return;
    }
    ESP_LOGI(TAG, "tensor_arena: %dKB @ PSRAM OK", TENSOR_ARENA_KB);

    ESP_LOGI(TAG, "Loading model...");
    const tflite::Model* model = tflite::GetModel(MODEL_DATA);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version mismatch!");
        return;
    }

    REGISTER_OPS(resolver);
    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, arena_size);
    g_interpreter = &interpreter;

    if (g_interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() thất bại!");
        return;
    }

    size_t used = g_interpreter->arena_used_bytes();
    ESP_LOGI(TAG, "Model OK | Arena used: %u / %d bytes (%.1f%%)",
             used, arena_size, 100.0f * used / arena_size);
    ESP_LOGI(TAG, "Input  : %dx%dx%d", IMG_W, IMG_H, IMG_CHANNELS);
    ESP_LOGI(TAG, "Classes: %d", NUM_CLASSES);

    if (!wifi_init_sta()) {
        ESP_LOGE(TAG, "Không có Wi-Fi — dừng.");
        return;
    }

    if (start_webserver() == NULL) {
        ESP_LOGE(TAG, "Không khởi động được HTTP server — dừng.");
        return;
    }

    ESP_LOGI(TAG, "✓ Sẵn sàng nhận ảnh qua POST /infer");
}