import cv2
import numpy as np

# Funções de efeito
def apply_gaussian_blur(frame, kernel_size):
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

def apply_canny_edge_detection(frame, low_threshold, high_threshold):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def apply_sobel(frame, ksize):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = np.uint8(magnitude)
    return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)

def adjust_brightness(frame, beta):
    return cv2.convertScaleAbs(frame, beta=beta)

def adjust_contrast(frame, alpha):
    return cv2.convertScaleAbs(frame, alpha=alpha)

def apply_negative_effect(frame):
    return cv2.bitwise_not(frame)

def convert_to_grayscalej(frame, i):
    r = 1 - (i - 1) / 29.0
    intensidade_mapeada = round(r, 3)

    # Aplica a interpolação de RGB para escala de tons de cinza
    frame_cinza = np.dot(frame[..., :3], [0.299, 0.587, 0.114])  # Ponderação de canais RGB

    # Aplica a interpolação
    frame_cinza_interpolado = frame_cinza + intensidade_mapeada * (255 - frame_cinza)

    return frame_cinza_interpolado.astype(np.uint8)

def convert_to_grayscale(frame, i):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Converta para BGR (formato de cores do OpenCV)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    r = 1 - (i - 1) / 29.0
    intensidade_mapeada = round(r, 3)

    # Ponderação entre o frame original e o cinza
    blended = cv2.addWeighted(frame, intensidade_mapeada, gray_bgr, (1 - intensidade_mapeada),0 )
                              
    return blended

def resize_video(frame):
    width = int(frame.shape[1] * 50 / 100)
    height = int(frame.shape[0] * 50 / 100)
    return cv2.resize(frame, (width, height))

def rotate_video(frame):
    return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

def mirror_video(frame, horizontal=True, vertical=True):
    if horizontal:
        frame = cv2.flip(frame, 1)
    if vertical:
        frame = cv2.flip(frame, 0)
    return frame  

def mirror_video_horizontal(frame):
    frame_espelhado = cv2.flip(frame, 1)
    return frame_espelhado 

# Inicialização da câmera
camera = 0
cap = cv2.VideoCapture(camera)

if not cap.isOpened():
    exit()

# Parametro barra de ajuste
bar = 0

# Função de callback para a barra de ajuste
def update_bar_adjustment(value):
    global bar
    bar = int(value)

# Criação da janela
cv2.namedWindow('Original Camera')
cv2.namedWindow('Camera with Effects')
cv2.createTrackbar('Value', 'Camera with Effects', 1, 30, update_bar_adjustment)

apply_gaussian_blur_flag = False
apply_canny_edge_detection_flag = False
apply_sobel_flag = False
apply_adjust_brightness_flag = False
apply_adjust_contrast_flag = False
apply_adjust_negative_flag = False
convert_to_grayscale_flag = False 
resize_video_flag = False
rotate_video_flag = False
mirror_video_flag = False
mirror_video_horizontal_flag = False
record = False

# Inicializa a câmera
cap = cv2.VideoCapture(0)

# Define o codec e cria o objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('video.avi', fourcc, 20.0, (640, 480))

while True:    
    ret, frame = cap.read()
    if not ret:
        break

    # Mostra a câmera original
    cv2.imshow('Original Camera', frame)
    # Obtém a tecla digitada pelo usuário
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    
    # Aplica os efeitos de acordo com a tecla
    if key == ord('L') or key == ord('l'):
        record = not record
    
    elif key == ord('A') or key == ord('a'):
        apply_gaussian_blur_flag = not apply_gaussian_blur_flag

    elif key == ord('B') or key == ord('b'):
        apply_canny_edge_detection_flag = not apply_canny_edge_detection_flag

    elif key == ord('C') or key == ord('c'):
        apply_sobel_flag = not apply_sobel_flag

    elif key == ord('D') or key == ord('d'):
        apply_adjust_brightness_flag = not apply_adjust_brightness_flag

    elif key == ord('E') or key == ord('e'):
        apply_adjust_contrast_flag = not apply_adjust_contrast_flag
    
    elif key == ord('F') or key == ord('f'):
       apply_adjust_negative_flag = not apply_adjust_negative_flag
    
    elif key == ord('G') or key == ord('g'):
        convert_to_grayscale_flag = not convert_to_grayscale_flag

    elif key == ord('H') or key == ord('h'):
        resize_video_flag = not resize_video_flag

    elif key == ord('I') or key == ord('i'):
        rotate_video_flag = not rotate_video_flag

    elif key == ord('J') or key == ord('j'):
        mirror_video_flag = not mirror_video_flag

    elif key == ord('K') or key == ord('k'):
        mirror_video_horizontal_flag = not mirror_video_horizontal_flag

    elif cv2.getWindowProperty('Original Camera', cv2.WND_PROP_VISIBLE) < 1 or cv2.getWindowProperty('Camera with Effects', cv2.WND_PROP_VISIBLE) < 1:
        break

    if apply_gaussian_blur_flag:
        value = 1
        if bar % 2 == 0:
            value = bar + 1
        else:
            value = bar
        frame = apply_gaussian_blur(frame, value)

    if apply_canny_edge_detection_flag:
        low_threshold = bar
        high_threshold = low_threshold * 3        
        frame = apply_canny_edge_detection(frame, low_threshold, high_threshold)

    if apply_sobel_flag:
        value = 1
        if bar % 2 == 0:
            value = bar + 1
        else:
            value = bar
        frame = apply_sobel(frame, value)

    if apply_adjust_brightness_flag:
        frame = adjust_brightness(frame, bar)

    if apply_adjust_contrast_flag:
        frame = adjust_contrast(frame, bar)

    if apply_adjust_negative_flag:
        frame = apply_negative_effect(frame)

    if convert_to_grayscale_flag:
        value = 1
        if bar % 2 == 0:
            value = bar + 1
        else:
            value = bar
        frame = convert_to_grayscale(frame, value)

    if resize_video_flag:
        frame = resize_video(frame)

    if rotate_video_flag:
        frame = rotate_video(frame)

    if mirror_video_flag:
        frame = mirror_video(frame, horizontal=True, vertical=True)

    if mirror_video_horizontal_flag:
        frame = mirror_video_horizontal(frame)

    if record:
        # Grava o frame no vídeo
        out.write(frame)

    # Mostra a câmera com efeitos
    cv2.imshow('Camera with Effects', frame)

cap.release()
cv2.destroyAllWindows()