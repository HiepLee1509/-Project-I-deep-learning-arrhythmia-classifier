import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from scipy.signal import find_peaks
import pywt # Thư viện Wavelet

# Classes tương ứng với 5 loại rối loạn nhịp tim
CLASS_INFO = {
    'N': {
        "name": "Bình thường (Normal)",
        "color": "green",
        "advice": "Nhịp tim của bạn đang ở trạng thái ổn định. Hãy duy trì lối sống lành mạnh, tập thể dục đều đặn và ăn uống cân bằng."
    },
    'S': {
        "name": "Ngoại tâm thu trên thất (SVEB)",
        "color": "orange",
        "advice": "Thường lành tính nhưng có thể do căng thẳng, caffeine hoặc thiếu ngủ. Nên hạn chế chất kích thích, nghỉ ngơi hợp lý. Nếu thấy hồi hộp nhiều, hãy đi khám."
    },
    'V': {
        "name": "Ngoại tâm thu thất (VEB)",
        "color": "red",
        "advice": "Có thể gây cảm giác hẫng nhịp. Nguyên nhân có thể do rối loạn điện giải, bệnh tim nền hoặc stress. Cần theo dõi tần suất, nếu xuất hiện dày đặc hoặc gây chóng mặt, cần gặp bác sĩ tim mạch ngay."
    },
    'F': {
        "name": "Nhịp hỗn hợp (Fusion Beat)",
        "color": "purple",
        "advice": "Là sự kết hợp giữa nhịp bình thường và nhịp bất thường. Đây là dấu hiệu cần được bác sĩ chuyên khoa đánh giá kỹ hơn qua Holter ECG."
    },
    'Q': {
        "name": "Nhịp không xác định (Unknown)",
        "color": "gray",
        "advice": "Tín hiệu bị nhiễu hoặc không rõ ràng. Vui lòng kiểm tra lại thiết bị đo, tiếp xúc điện cực và đo lại trong trạng thái tĩnh. Hoặc đi khám chuyên khoa để được đánh giá chính xác hơn."
    }
}

CLASSES_KEYS = ['N', 'S', 'V', 'F', 'Q']

def load_arrhythmia_model(model_path="model\\ecg_model_code 17_t5.h5"):
    try:
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        print(f"Lỗi tải model: {e}")
        return None

def get_model_input_length(model):
    """Tự động lấy độ dài input đầu vào của model"""
    try:
        input_shape = model.input_shape
        if input_shape and len(input_shape) >= 2 and input_shape[1] is not None:
            return int(input_shape[1])      #input_shape = (None, 187, 1) = (batch_size, input_length, features)
        # Nếu không lấy được từ input_shape, thử lấy từ lớp đầu tiên
        first_layer = model.layers[0]
        if hasattr(first_layer, 'input_shape'):
            cfg_shape = first_layer.input_shape
            if cfg_shape and len(cfg_shape) >= 2 and cfg_shape[1] is not None:
                 return int(cfg_shape[1])
    except:
        pass
    return 187 # Fallback

def denoise_signal_wavelet(signal, wavelet='sym8', level=1):
    """Lọc nhiễu Wavelet"""
    if len(signal) < 10:
        return signal
    try:
        coeffs = pywt.wavedec(signal, wavelet, mode='per', level=level) 
        detail_coeffs = coeffs[-1]
        sigma = np.median(np.abs(detail_coeffs)) / 0.6745 
        thresh = sigma * np.sqrt(2 * np.log(len(signal)))
        new_coeffs = [coeffs[0]]
        for c in coeffs[1:]:
            new_coeffs.append(pywt.threshold(c, thresh, mode='soft'))
        denoised_signal = pywt.waverec(new_coeffs, wavelet, mode='per')
        
        if len(denoised_signal) > len(signal):
            denoised_signal = denoised_signal[:len(signal)]
        elif len(denoised_signal) < len(signal):
            pad_width = len(signal) - len(denoised_signal)
            denoised_signal = np.pad(denoised_signal, (0, pad_width), 'edge')
        return denoised_signal
    except:
        return signal

def detect_and_segment(denoised_ecg_signal, r_peak_height=0.5, r_peak_distance=150, output_length=187):
    """Phát hiện đỉnh R và phân đoạn"""
    peaks, _ = find_peaks(denoised_ecg_signal, height=r_peak_height, distance=r_peak_distance)
    
    ratio_before = 99 / 187
    window_before = int(output_length * ratio_before)
    window_after = output_length - window_before - 1
    
    segments = []
    valid_peak_locations = []
    
    for peak_loc in peaks:
        start = peak_loc - window_before
        end = peak_loc + window_after + 1
        if start < 0 or end > len(denoised_ecg_signal):
            continue
        segment = denoised_ecg_signal[start : end]
        if len(segment) == output_length:
            segments.append(segment)
            valid_peak_locations.append(peak_loc)
        
    if not segments:
        return np.array([]), np.array([])
        
    return np.array(segments), np.array(valid_peak_locations)

def predict_from_segments(segments_array, model):
    """Dự đoán và trả về mã lớp (N, S, V...)"""
    if segments_array.ndim == 2:
        X = segments_array.reshape(-1, segments_array.shape[1], 1)  # Thêm chiều features=1
    else:
        X = segments_array

    y_pred_probs = model.predict(X)
    y_pred_indices = np.argmax(y_pred_probs, axis=1)
    
    # Trả về mã ký tự (N, S, V...) để frontend tra cứu trong CLASS_INFO
    predicted_codes = [CLASSES_KEYS[i] for i in y_pred_indices]
    return predicted_codes, y_pred_indices

def calculate_hrv_metrics(peaks, fs=360):
    """
    Tính toán các chỉ số biến thiên nhịp tim (HRV) cơ bản.
    Input:
        peaks: mảng chứa vị trí (index) các đỉnh R
        fs: tần số lấy mẫu
    Output:
        dict chứa các chỉ số và dữ liệu vẽ biểu đồ
    """
    if len(peaks) < 2:
        return None
    
    # 1. Tính khoảng cách RR (RR intervals) ra đơn vị mili-giây (ms)
    # np.diff(peaks) là khoảng cách giữa các đỉnh liên tiếp (tính bằng số mẫu)
    rr_intervals = np.diff(peaks)
    rr_ms = (rr_intervals / fs) * 1000
    
    # 2. Tính các chỉ số HRV (Time-domain)
    # SDNN: Độ lệch chuẩn của các khoảng RR (Đánh giá sức khỏe tổng quát)
    sdnn = np.std(rr_ms)
    
    # RMSSD: Căn bậc hai của trung bình bình phương sự khác biệt giữa các khoảng RR liên tiếp
    # (Đánh giá hoạt động của hệ thần kinh phó giao cảm)
    diff_rr = np.diff(rr_ms)
    rmssd = np.sqrt(np.mean(diff_rr**2))
    
    # Nhịp tim trung bình (BPM)
    mean_rr = np.mean(rr_ms)
    mean_bpm = 60000 / mean_rr
    
    # 3. Chuẩn bị dữ liệu Poincaré Plot
    # Trục X: RR[n], Trục Y: RR[n+1]
    poincare_x = rr_ms[:-1]
    poincare_y = rr_ms[1:]
    
    return {
        "rr_ms": rr_ms,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "mean_bpm": mean_bpm,
        "poincare_x": poincare_x,
        "poincare_y": poincare_y
    }

def analyze_batch_data(patient_data_map, model, fs=360, wavelet='sym8', r_peak_height=0.5):
    """
    Chạy phân tích hàng loạt trên toàn bộ dataset.
    Trả về DataFrame tóm tắt để hiển thị bảng.
    """
    results = []
    
    # Lấy độ dài input cần thiết
    required_len = get_model_input_length(model)
    
    # Duyệt qua từng bệnh nhân/bản ghi
    # Sử dụng enumerate để trả về tiến trình nếu cần
    total_files = len(patient_data_map)
    
    for idx, (pid, raw_signal) in enumerate(patient_data_map.items()):
        try:
            # 1. Chuyển đổi sang numpy array
            signal = np.array(raw_signal)
            
            # 2. Xử lý tín hiệu
            denoised = denoise_signal_wavelet(signal, wavelet=wavelet)
            segments, peaks = detect_and_segment(denoised, r_peak_height, output_length=required_len)
            
            stats = {
                "ID": pid,
                "Total Beats": 0,
                "BPM (Avg)": 0,
                "Status": "Error",
                "Risk Level": "Unknown",
                "N": 0, "S": 0, "V": 0, "F": 0, "Q": 0
            }

            if len(segments) > 0:
                # 3. Dự đoán
                pred_codes, _ = predict_from_segments(segments, model)
                
                # 4. Thống kê
                counts = pd.Series(pred_codes).value_counts()
                total_beats = len(pred_codes)
                
                # Tính nhịp tim trung bình
                if len(peaks) > 1:
                    avg_diff = np.mean(np.diff(peaks))
                    bpm = int(60 / (avg_diff / fs))
                else:
                    bpm = 0
                
                # Cập nhật stats
                stats["Total Beats"] = total_beats
                stats["BPM (Avg)"] = bpm
                stats["Status"] = "Success"
                
                # Fill số lượng từng loại
                for code in ['N', 'S', 'V', 'F', 'Q']:
                    count = counts.get(code, 0)
                    stats[code] = count
                
                # Đánh giá mức độ nguy hiểm
                if stats['V'] > 0 or stats['F'] > 0:
                    stats['Risk Level'] = "High 🔴"
                elif stats['S'] > 0:
                    stats['Risk Level'] = "Medium 🟡"
                else:
                    stats['Risk Level'] = "Low 🟢"
            else:
                stats["Status"] = "No Peaks Found"
                
            results.append(stats)
            
        except Exception as e:
            results.append({"ID": pid, "Status": f"Error: {str(e)}", "Risk Level": "Error"})

    return pd.DataFrame(results)

def generate_ai_doctor_advice(batch_df):
    """
    Phân tích kết quả quét hàng loạt và tạo lời khuyên từ AI Doctor.
    
    Args:
        batch_df: DataFrame chứa kết quả phân tích từ analyze_batch_data
        
    Returns:
        dict chứa thông tin advice: {
            'level': 'excellent' | 'warning' | 'danger' | 'caution',
            'title': str,
            'message': str,
            'recommendations': list of str,
            'icon': str
        }
    """
    if batch_df is None or len(batch_df) == 0:
        return {
            'level': 'info',
            'title': 'Chưa có dữ liệu',
            'message': 'Vui lòng quét dữ liệu trước khi xem lời khuyên.',
            'recommendations': [],
            'icon': 'ℹ️'
        }
    
    # Tính tổng số nhịp theo từng loại
    total_beats = batch_df[['N', 'S', 'V', 'F', 'Q']].sum()
    total_all_beats = total_beats.sum()
    
    if total_all_beats == 0:
        return {
            'level': 'warning',
            'title': 'Không phát hiện nhịp tim',
            'message': 'Không tìm thấy nhịp tim hợp lệ trong dữ liệu. Vui lòng kiểm tra lại chất lượng tín hiệu ECG.',
            'recommendations': [
                'Kiểm tra lại thiết bị đo ECG',
                'Đảm bảo điện cực tiếp xúc tốt với da',
                'Thử đo lại trong môi trường yên tĩnh'
            ],
            'icon': '⚠️'
        }
    
    # Tính phần trăm từng loại
    pct_N = (total_beats['N'] / total_all_beats) * 100
    pct_S = (total_beats['S'] / total_all_beats) * 100
    pct_V = (total_beats['V'] / total_all_beats) * 100
    pct_F = (total_beats['F'] / total_all_beats) * 100
    pct_Q = (total_beats['Q'] / total_all_beats) * 100
    
    # Đếm số ca có nguy cơ cao
    high_risk_count = len(batch_df[batch_df['Risk Level'].str.contains("High", na=False)])
    medium_risk_count = len(batch_df[batch_df['Risk Level'].str.contains("Medium", na=False)])
    total_patients = len(batch_df)
    high_risk_pct = (high_risk_count / total_patients) * 100 if total_patients > 0 else 0
    
    # Logic phân tích và đưa ra lời khuyên
    if pct_N > 95:
        # Excellent Health - >95% Normal
        return {
            'level': 'excellent',
            'title': 'Sức khỏe tim mạch xuất sắc',
            'message': f'Kết quả phân tích cho thấy {pct_N:.1f}% nhịp tim là bình thường. Đây là dấu hiệu rất tích cực về sức khỏe tim mạch của bạn.',
            'recommendations': [
                'Tiếp tục duy trì lối sống lành mạnh',
                'Tập thể dục đều đặn ít nhất 30 phút mỗi ngày',
                'Ăn uống cân bằng, hạn chế chất béo và muối',
                'Khám sức khỏe định kỳ 6 tháng/lần',
                'Quản lý căng thẳng và ngủ đủ giấc'
            ],
            'icon': '✅',
            'stats': {
                'normal_pct': pct_N,
                'total_beats': int(total_all_beats),
                'total_patients': total_patients
            }
        }
    
    elif pct_V > 5 or pct_F > 2 or high_risk_pct > 20:
        # Danger - High frequency of VEB/Fusion or many high-risk patients
        return {
            'level': 'danger',
            'title': 'Cảnh báo: Phát hiện rối loạn nhịp tim nghiêm trọng',
            'message': f'Phát hiện {pct_V:.1f}% nhịp ngoại tâm thu thất (VEB) và {pct_F:.1f}% nhịp hỗn hợp. {high_risk_pct:.1f}% số ca có nguy cơ cao. Đây là dấu hiệu cần được đánh giá y tế ngay lập tức.',
            'recommendations': [
                '⚠️ KHẨN CẤP: Liên hệ bác sĩ tim mạch trong vòng 24-48 giờ',
                'Tránh các hoạt động gắng sức cho đến khi được đánh giá',
                'Theo dõi các triệu chứng: đau ngực, khó thở, chóng mặt',
                'Gọi cấp cứu 115 nếu xuất hiện đau ngực dữ dội hoặc ngất xỉu',
                'Chuẩn bị hồ sơ y tế và kết quả ECG này để bác sĩ xem xét',
                'Tránh caffeine, rượu và các chất kích thích'
            ],
            'icon': '🚨',
            'stats': {
                'veb_pct': pct_V,
                'fusion_pct': pct_F,
                'high_risk_pct': high_risk_pct,
                'total_beats': int(total_all_beats)
            }
        }
    
    elif pct_S > 10 or medium_risk_count > total_patients * 0.3:
        # Warning - High frequency of SVEB
        return {
            'level': 'warning',
            'title': 'Cảnh báo: Phát hiện rối loạn nhịp tim nhẹ',
            'message': f'Phát hiện {pct_S:.1f}% nhịp ngoại tâm thu trên thất (SVEB). Mặc dù thường lành tính, nhưng tần suất cao có thể cần được theo dõi.',
            'recommendations': [
                'Nên đi khám bác sĩ tim mạch trong vòng 1-2 tuần',
                'Theo dõi các triệu chứng: hồi hộp, đánh trống ngực',
                'Hạn chế caffeine, rượu và các chất kích thích',
                'Quản lý căng thẳng và đảm bảo ngủ đủ giấc',
                'Tập thể dục nhẹ nhàng, tránh gắng sức quá mức',
                'Ghi nhật ký các triệu chứng để báo cáo với bác sĩ'
            ],
            'icon': '⚠️',
            'stats': {
                'sveb_pct': pct_S,
                'medium_risk_count': medium_risk_count,
                'total_beats': int(total_all_beats)
            }
        }
    
    elif pct_N < 80:
        # Caution - Lower than expected normal beats
        return {
            'level': 'caution',
            'title': 'Lưu ý: Cần theo dõi thêm',
            'message': f'Chỉ có {pct_N:.1f}% nhịp tim bình thường. Mặc dù không có dấu hiệu nguy hiểm ngay lập tức, nhưng nên được đánh giá thêm.',
            'recommendations': [
                'Nên đi khám bác sĩ để được đánh giá toàn diện',
                'Theo dõi các triệu chứng bất thường',
                'Duy trì lối sống lành mạnh',
                'Tránh các yếu tố gây căng thẳng',
                'Cân nhắc đo Holter ECG 24h để theo dõi liên tục'
            ],
            'icon': '💡',
            'stats': {
                'normal_pct': pct_N,
                'total_beats': int(total_all_beats)
            }
        }
    
    else:
        # Good but not excellent
        return {
            'level': 'good',
            'title': 'Sức khỏe tim mạch tốt',
            'message': f'Kết quả phân tích cho thấy {pct_N:.1f}% nhịp tim bình thường. Sức khỏe tim mạch của bạn đang ở mức tốt.',
            'recommendations': [
                'Tiếp tục duy trì lối sống lành mạnh',
                'Tập thể dục đều đặn',
                'Khám sức khỏe định kỳ',
                'Theo dõi các chỉ số tim mạch'
            ],
            'icon': '👍',
            'stats': {
                'normal_pct': pct_N,
                'total_beats': int(total_all_beats)
            }
        }