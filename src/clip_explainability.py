import torch
import CLIP.clip as clip
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from captum.attr import visualization
from torchvision.transforms import ToTensor
import sgolay2.sgolay2 as sgolay2


def AttentionMap(images, texts, model, preprocess, device):
    '''
    image: PIL Image
    texts: list of str
    '''
    image = preprocess(images).unsqueeze(0).to(device)
    texts = clip.tokenize(texts).to(device)

    batch_size = texts.shape[0]
    
    logits_per_image, logits_per_text = model(image, texts)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

    start_layer = len(image_attn_blocks) - 1
    
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]

    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
          continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    image_relevance = R[:, 0, 1:]

    
    # text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())

    # start_layer_text = len(text_attn_blocks) - 1

    # num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
    # R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(device)
    # R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    # for i, blk in enumerate(text_attn_blocks):
    #     if i < start_layer_text:
    #       continue
    #     grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
    #     cam = blk.attn_probs.detach()
    #     cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
    #     grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
    #     cam = grad * cam
    #     cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
    #     cam = cam.clamp(min=0).mean(dim=1)
    #     R_text = R_text + torch.bmm(cam, R_text)
    # text_relevance = R_text
    
   
    return image_relevance, image

def AttentionMapwithImage(image_relevance, image):
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=(image.shape[-2:]), mode='bilinear')
    image_relevance = image_relevance.reshape(image.shape[-2:]).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image = image.permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def fitandsmoothing(image_relevance, image):
    # image_relevance : [1,dim,dim]
    # image : [3,H,W]
    print(image_relevance.shape, image.shape)
    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    print(image_relevance)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=(image.shape[-2:]), mode='bilinear')
    image_relevance = image_relevance.reshape(image.shape[-2:]).cuda().data.clone().detach().to(torch.float32).cpu().numpy()
    image_relevance = ((image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min()))
    print(type(image_relevance))
    smooth_image_relevance = sgolay2.SGolayFilter2(window_size=31, poly_order=3)(image_relevance.squeeze())
    # fig, ax = plt.subplots(1,2)
    # # ax = fig.add_subplot(111, projection='3d')
    # x = torch.arange(smooth_image_relevance.shape[0])
    # y = torch.arange(smooth_image_relevance.shape[1])
    # x, y = torch.meshgrid(x, y)
    # # ax.plot_wireframe(x, y, smooth_image_relevance, linewidths=0.5, color='r')
    # # ax.scatter(x, y, smooth_image_relevance, s=5, c='r')

    # # ax.plot_surface(x, y, smooth_image_relevance, linewidth=0)
    # # ax.plot_surface(x, y, image_relevance, color='y', linewidth=0, alpha=0.4)
    # ax[1].imshow(smooth_image_relevance)
    # ax[0].imshow(image_relevance)
        
    # plt.show()
    
    return smooth_image_relevance

from skimage.feature import peak_local_max

def find_peaks(attention_map, min_distance_ratio=0.05, threshold_abs=0.3):
    """
    attention_map: 2D numpy array (H, W)
    min_distance: 두 peak 사이의 최소 거리(픽셀)
    threshold_abs: 이 값 이상인 곳만 peak 후보로 간주
    """
    min_distance = int((attention_map.shape[-2]+attention_map.shape[-1])/2*min_distance_ratio)
    print(min_distance)
    # peak_local_max는 반환값으로 (row, col) 좌표 리스트를 준다.
    peaks = peak_local_max(attention_map, 
                           min_distance=min_distance, 
                           threshold_abs=threshold_abs)
    return peaks  # shape = (num_peaks, 2)

import numpy as np
import cv2

def extract_rois_from_attention_map(attention_map, threshold_ratio=0.4):
    """
    attention_map: shape = (H, W), 0~1 사이 값이라고 가정
    threshold: 0~1 사이. threshold 이상인 픽셀들만 ROI로 설정
    """
    # 1) Thresholding
    #    threshold 이 넘는 부분만 1로 만들어서 관심 영역 후보 마스크 생성
    threshold = attention_map.max() * threshold_ratio
    print(attention_map.max(), attention_map.min())
    
    bin_map = (attention_map >= threshold).astype(np.uint8)
    plt.imshow(bin_map)
    plt.show()
    print(bin_map)
    # 2) Connected Components
    #    각 관심 영역(connected component)에 대한 레이블, 통계치 추출
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_map, connectivity=4)
    # stats: shape = [num_labels, 5], 각각 [left, top, width, height, area]
    # 첫 번째 label = 0 은 배경(background) 정보
    # 관심 영역에 대한 bbox와 스코어를 저장할 리스트
    rois = []
    for label_idx in range(1, num_labels):  # 0은 background이므로 건너뜀
        left   = stats[label_idx, cv2.CC_STAT_LEFT]
        top    = stats[label_idx, cv2.CC_STAT_TOP]
        width  = stats[label_idx, cv2.CC_STAT_WIDTH]
        height = stats[label_idx, cv2.CC_STAT_HEIGHT]

        # 이 connected component에 해당하는 attention 맵 값들만 추출
        region = attention_map[top: top + height, left: left + width]

        # 3) ROI 스코어 계산
        #    예: 해당 region의 합(sum), 평균(mean), 최댓값(max) 등 다양하게 가능
        score = region.max()

        # 결과 저장
        rois.append({
            'bbox': (left, top, width, height), 
            'score': score,
            'label_idx': label_idx,
        })

    # 4) score 기준 내림차순 정렬
    rois.sort(key=lambda x: x['score'], reverse=True)

    return rois

import torchvision

def visualize_rois_in_image(image, rois, max_rois=5):
    """
    image: NumPy array of shape (H, W, 3), BGR 혹은 RGB
    rois: extract_rois_from_attention_map()의 리턴값
    max_rois: 상위 몇 개까지 그릴지
    """
    # 시각화를 위해 복제
    vis_image = image.copy()
    max_rois = min(max_rois, len(rois))
    
    for i, roi in enumerate(rois[:max_rois]):
        left, top, width, height = roi['bbox']
        score = roi['score']

        # ROI 그리기 (파랑 테두리: (255,0,0))
        cv2.rectangle(vis_image, (left, top), (left+width, top+height), (255, 0, 0), 2)
        # ROI 순번과 점수 표시
        cv2.putText(vis_image, f"{i+1}:{score:.2f}", (left, top-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    return vis_image

def compute_gradient(tensor, delta_x=1.0, delta_y=1.0):
    """
    (H, W) 크기의 텐서에 대해 x와 y 방향의 그라디언트를 계산하여 (H, W, 2) 형태로 반환합니다.
    
    Parameters:
    - tensor: 2D numpy 배열, 형태는 (H, W)
    - delta_x: x 방향(열 방향) 격자 간격
    - delta_y: y 방향(행 방향) 격자 간격
    
    Returns:
    - gradient: 3D numpy 배열, 형태는 (H, W, 2)
                gradient[:,:,0] -> x 방향 그라디언트
                gradient[:,:,1] -> y 방향 그라디언트
    """
    H, W = tensor.shape
    gradient = np.zeros((H, W, 2), dtype=np.float64)
    
    # x 방향 그라디언트 (열 방향)
    # 중앙차분
    gradient[1:-1, :, 0] = (tensor[2:, :] - tensor[:-2, :]) / (2 * delta_x)
    # 경계점 - 순방향 차분 (왼쪽 경계)
    gradient[0, :, 0] = (tensor[1, :] - tensor[0, :]) / delta_x
    # 경계점 - 역방향 차분 (오른쪽 경계)
    gradient[-1, :, 0] = (tensor[-1, :] - tensor[-2, :]) / delta_x
    
    # y 방향 그라디언트 (행 방향)
    # 중앙차분
    gradient[:, 1:-1, 1] = (tensor[:, 2:] - tensor[:, :-2]) / (2 * delta_y)
    # 경계점 - 순방향 차분 (위쪽 경계)
    gradient[:, 0, 1] = (tensor[:, 1] - tensor[:, 0]) / delta_y
    # 경계점 - 역방향 차분 (아래쪽 경계)
    gradient[:, -1, 1] = (tensor[:, -1] - tensor[:, -2]) / delta_y
    
    return gradient

def plot_vector_field(gradient):
    """
    2D 벡터장을 quiver plot으로 시각화합니다.
    
    Parameters:
    - gradient: (H, W, 2) 형태의 numpy 배열
                gradient[:,:,0]는 x 방향 (열 방향) 성분
                gradient[:,:,1]는 y 방향 (행 방향) 성분
    """
    H, W, _ = gradient.shape
    
    # 격자점 생성
    x = np.arange(W)
    y = np.arange(H)
    X, Y = np.meshgrid(x, y)  # (H, W) 형태의 격자 좌표
    
    # 벡터 필드의 x, y 성분
    U = gradient[:, :, 0]  # x 방향 (열 방향) 성분
    V = gradient[:, :, 1]  # y 방향 (행 방향) 성분
    
    # Quiver plot
    plt.figure(figsize=(8, 6))
    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='blue')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('2D Vector Field')
    plt.gca().invert_yaxis()  # 그래프를 일반적인 (0,0) 좌표에서 위로 커지도록
    plt.grid(True)
    plt.axis('equal')  # 축 비율 고정
    plt.show()
    
from scipy.ndimage import label

def watershed(sir, threshold_ratio=0.1):
    sir = 1-sir
    sir *= 255
    sir = np.expand_dims(sir,axis=2)
    sir = sir.astype(np.uint8)
    ret, thr = cv2.threshold(sir, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=2)
    
    border = cv2.dilate(opening, kernel, iterations=3)
    border = border - cv2.erode(border, None)
    
    dt = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    dt = ((dt-dt.min())/(dt.max()-dt.min())*255).astype(np.uint8)
    ret, dt = cv2.threshold(dt, 180, 255, cv2.THRESH_BINARY)
    
    marker, ncc = label(dt)
    marker = marker*(255/ncc)
    
    marker[border==255] = 255
    marker = marker.astype(np.int32)
    
    marker[marker==-1] = 0
    marker = marker.astype(np.uint8)
    marker = 255 - marker
    
    marker[marker!=255] = 0
    marker = cv2.dilate(marker, None)
    
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(sir)
    ax[1].imshow(marker)
    plt.show()

clip.clip._MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}
class CE:
    def __init__(self, args):
        self.device = args.device
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        self.spot_size = args.spot_size
    def getLoc(self, image, query):
        image_relevance,_ = AttentionMap(image,[query],self.model,self.preprocess,self.device)
        sir = fitandsmoothing(image_relevance, ToTensor()(image))
        peaks = find_peaks(sir,0.05,0.3)
        answer = np.zeros((peaks.shape[0],4))
        for i in range(len(peaks)):
            answer[i] = [peaks[i][1]-int(self.spot_size*image.size[0]*0.5),peaks[i][0]-int(self.spot_size*image.size[1]*0.5),int(self.spot_size*image.size[0]),int(self.spot_size*image.size[1])]
            answer[i] = np.array(list(map(lambda x:max(x,0),answer[i])))
        return answer
    

if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.device="cuda:0"
            self.spot_size=0.2
    myce = CE(Args())
    print(myce.getLoc(Image.open(r"C:\Users\user\Downloads\image1.png"),"how many creatures in this image?"))