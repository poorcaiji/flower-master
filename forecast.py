import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model

# 全局变量，用于缓存模型和类别信息
_model = None
_classes = None
_class_indices = None

# 花卉名称映射表
FLOWER_NAMES = {
    '0': 'pink primrose',
    '1': 'hard-leaved pocket orchid',
    '2': 'canterbury bells',
    '3': 'sweet pea',
    '4': 'english marigold',
    '5': 'tiger lily',
    '6': 'moon orchid',
    '7': 'bird of paradise',
    '8': 'monkshood',
    '9': 'globe thistle',
    '10': 'snapdragon',
    '11': "colt's foot",
    '12': 'king protea',
    '13': 'spear thistle',
    '14': 'yellow iris',
    '15': 'globe-flower',
    '16': 'purple coneflower',
    '17': 'peruvian lily',
    '18': 'balloon flower',
    '19': 'giant white arum lily',
    '20': 'fire lily',
    '21': 'pincushion flower',
    '22': 'fritillary',
    '23': 'red ginger',
    '24': 'grape hyacinth',
    '25': 'corn poppy',
    '26': 'prince of wales feathers',
    '27': 'stemless gentian',
    '28': 'artichoke',
    '29': 'sweet william',
    '30': 'carnation',
    '31': 'garden phlox',
    '32': 'love in the mist',
    '33': 'mexican aster',
    '34': 'alpine sea holly',
    '35': 'ruby-lipped cattleya',
    '36': 'cape flower',
    '37': 'great masterwort',
    '38': 'siam tulip',
    '39': 'lenten rose',
    '40': 'barbeton daisy',
    '41': 'daffodil',
    '42': 'sword lily',
    '43': 'poinsettia',
    '44': 'bolero deep blue',
    '45': 'wallflower',
    '46': 'marigold',
    '47': 'buttercup',
    '48': 'oxeye daisy',
    '49': 'common dandelion',
    '50': 'petunia',
    '51': 'wild pansy',
    '52': 'primula',
    '53': 'sunflower',
    '54': 'pelargonium',
    '55': 'bishop of llandaff',
    '56': 'gaura',
    '57': 'geranium',
    '58': 'orange dahlia',
    '59': 'pink-yellow dahlia',
    '60': 'cautleya spicata',
    '61': 'japanese anemone',
    '62': 'black-eyed susan',
    '63': 'silverbush',
    '64': 'californian poppy',
    '65': 'osteospermum',
    '66': 'spring crocus',
    '67': 'bearded iris',
    '68': 'windflower',
    '69': 'tree poppy',
    '70': 'gazania',
    '71': 'azalea',
    '72': 'water lily',
    '73': 'rose',
    '74': 'thorn apple',
    '75': 'morning glory',
    '76': 'passion flower',
    '77': 'lotus',
    '78': 'toad lily',
    '79': 'anthurium',
    '80': 'frangipani',
    '81': 'clematis',
    '82': 'hibiscus',
    '83': 'columbine',
    '84': 'desert-rose',
    '85': 'tree mallow',
    '86': 'magnolia',
    '87': 'cyclamen',
    '88': 'watercress',
    '89': 'canna lily',
    '90': 'hippeastrum',
    '91': 'bee balm',
    '92': 'ball moss',
    '93': 'foxglove',
    '94': 'bougainvillea',
    '95': 'camellia',
    '96': 'mallow',
    '97': 'mexican petunia',
    '98': 'bromelia',
    '99': 'blanket flower',
    '100': 'trumpet creeper',
    '101': 'blackberry lily'
}

# 花卉中文名称映射表
FLOWER_NAMES_CN = {
    '0':   '粉红报春花',
    '1':   '硬叶袋兰花',
    '2':   '坎特伯雷钟花',
    '3':   '香豌豆',
    '4':   '英国万寿菊',
    '5':   '虎百合',
    '6':   '月亮兰花',
    '7':   '天堂鸟',
    '8':   '乌头花',
    '9':   '球状蓟',
    '10':  '金鱼草',
    '11':  '款冬花',
    '12':  '帝王普罗梯亚',
    '13':  '长刺蓟',
    '14':  '黄鸢尾',
    '15':  '球花',
    '16':  '紫锥花',
    '17':  '秘鲁百合',
    '18':  '桔梗',
    '19':  '巨型白色天南星',
    '20':  '火百合',
    '21':  '针垫花',
    '22':  '贝母花',
    '23':  '红姜花',
    '24':  '葡萄风信子',
    '25':  '虞美人',
    '26':  '威尔士王子羽毛',
    '27':  '无茎龙胆',
    '28':  '洋蓟',
    '29':  '甜威廉',
    '30':  '康乃馨',
    '31':  '花园福禄考',
    '32':  '雾中爱',
    '33':  '墨西哥翠菊',
    '34':  '高山海冬青',
    '35':  '红唇卡特兰',
    '36':  '好望角花',
    '37':  '大师草',
    '38':  '暹罗郁金香',
    '39':  '四旬花',
    '40':  '巴氏菊花',
    '41':  '水仙花',
    '42':  '剑兰',
    '43':  '一品红',
    '44':  '深蓝波列罗',
    '45':  '桂竹香',
    '46':  '万寿菊',
    '47':  '毛茛',
    '48':  '牛眼菊',
    '49':  '蒲公英',
    '50':  '矮牵牛',
    '51':  '野生三色堇',
    '52':  '报春花',
    '53':  '向日葵',
    '54':  '天竺葵',
    '55':  '兰达夫主教',
    '56':  '嘉宝花',
    '57':  '天竺葵',
    '58':  '橙色大丽花',
    '59':  '粉黄大丽花',
    '60':  '距花姜',
    '61':  '日本银莲花',
    '62':  '黑眼苏珊',
    '63':  '银叶',
    '64':  '加州罂粟',
    '65':  '南非菊',
    '66':  '春番红花',
    '67':  '有须鸢尾',
    '68':  '银莲花',
    '69':  '树罂粟',
    '70':  '勋章菊',
    '71':  '杜鹃花',
    '72':  '睡莲',
    '73':  '玫瑰',
    '74':  '曼陀罗',
    '75':  '牵牛花',
    '76':  '西番莲',
    '77':  '莲花',
    '78':  '蟾蜍百合',
    '79':  '火鹤花',
    '80':  '缅栀子花',
    '81':  '铁线莲',
    '82':  '木槿',
    '83':  '耧斗菜',
    '84':  '沙漠玫瑰',
    '85':  '树锦葵',
    '86':  '木兰',
    '87':  '仙客来',
    '88':  '豆瓣菜',
    '89':  '美人蕉',
    '90':  '朱顶红',
    '91':  '蜜蜂花',
    '92':  '球藓',
    '93':  '毛地黄',
    '94':  '三角梅',
    '95':  '山茶花',
    '96':  '锦葵',
    '97':  '墨西哥矮牵牛',
    '98':  '凤梨',
    '99':  '天人菊',
    '100': '凌霄花',
    '101': '黑莓百合'
}

def _load_model_and_classes():
    """
    加载模型和类别，使用全局变量缓存
    """
    global _model, _classes, _class_indices
    
    if _model is None:
        print("首次加载模型...")
        # 获取花卉数据类别
        test_dir = "./data/test"
        _classes = sorted([o for o in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, o))], key=int)
        
        # 加载模型
        _model = load_model("D:\\code\\pythonProject1\\pythonProject\\flower-master\\other_model\\model-resnet50.h5")
        
        # 预先计算类别索引字典
        _class_indices = dict(zip(_classes, range(len(_classes))))
        
        # 预热模型
        dummy_input = np.zeros((1, 224, 224, 3))
        _model.predict(dummy_input)
    
    return _model, _classes, _class_indices

def predict_flower(image_path):
    """
    对花卉图片进行预测
    
    Args:
        image_path: 图片路径
        
    Returns:
        list: 预测结果列表，每个元素为字典 {'class': 类别, 'probability': 概率, 'name': 花卉名称, 'name_cn': 花卉中文名称}
    """
    try:
        # 加载模型和类别（如果尚未加载）
        model, classes, class_indices = _load_model_and_classes()
        
        # 图片预处理
        image_size = (224, 224)
        img = image.load_img(image_path, target_size=image_size)
        img_array = np.expand_dims(image.img_to_array(img), axis=0)
        prepared_img = preprocess_input(img_array)
        
        # 预测
        out = model.predict(prepared_img, verbose=0)  # 禁用进度条

        top10 = out[0].argsort()[-10:][::-1]

        keys = list(class_indices.keys())
        values = list(class_indices.values())

        results = []
        for i, t in enumerate(top10):
            class_name = keys[values.index(t)]
            probability = float(out[0][t])
            flower_name = FLOWER_NAMES.get(class_name, f"未知花卉-{class_name}")
            flower_name_cn = FLOWER_NAMES_CN.get(class_name, "未知花卉")
            results.append({
                'class': class_name,
                'probability': probability,
                'name': flower_name,
                'name_cn': flower_name_cn
            })
            
        return results
    
    except Exception as e:
        raise Exception(f"预测过程中发生错误: {str(e)}")

# 如果直接运行这个脚本，则使用默认图片进行预测
if __name__ == '__main__':
    # 需要预测的图片的位置
    predict_image_path = "./data/test/2/image_06642.jpg"
    
    results = predict_flower(predict_image_path)
    
    print("Top10 的分类及概率：")
    for result in results:
        print("class:", result['class'], "name:", result['name'], "中文名:", result['name_cn'], "probability:", result['probability'])