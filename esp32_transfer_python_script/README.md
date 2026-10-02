## Scripts dùng để truyền ảnh sang ESP32 để thực hiện dự doán và nhận lại kết quả
1. Clone script này về máy
```
git clone https://github.com/vietdai-bk/esp32_transfer_python_script
```
2. Cài đặt các thư viện
```
pip install requests opencv-python
```
3. Cập nhật địa chỉ IP được log ra từ ESP32 ở file ```send_one_image.py```
```
ESP32_IP = "192.168.1.6"
```
4. Chạy test với một ảnh
```
python send_one_image.py /path/to/image.png
```
5. Chạy với một folder ảnh (thực tế tại Vòng Chung Kết)  
***Cập nhật lại đường dẫn tới folder, ip của esp32***, thực hiện chạy lệnh sau:
```
python main.py
```
Kết quả lưu tại ```results.csv```, các đội thi nộp lại file này về cho BTC tại ngày chung kết để tính điểm.  
### Lưu ý
Đầu vào của mô hình được deploy trên ESP32 ở định dạng uint8, nên các đội khi train nên tránh chuẩn hóa (x - mean) / std.  
Sau vòng loại BTC sẽ cung cấp đáp án của vòng loại, các đội thi nên test thử với tập test và so sánh kết quả file xuất ra với kết quả chính thức để đảm bảo mô hình hoạt động đúng. (Mô hình dự đoán sai có thể do cả cách chuẩn hóa dữ liệu đầu vào...)
