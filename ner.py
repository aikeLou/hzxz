from paddlenlp import Taskflow
import re, bisect, json, jionlp
from collections import Counter, OrderedDict, defaultdict
####
KEYWORD = OrderedDict()
KEYWORD['人员角色_煽动者'] = [
    '煽惑', '倡议', '号召', '煽动', '鼓舞', '挑唆', '率先', '鼓动', '起初', '发起', '挑头', '第一',
    '引领', '带头', '领头', '激励', '首先'
]
KEYWORD['人员角色_响应者'] = ['回应', '响应', '回答', '反应', '回响', '回覆', '回复']
KEYWORD['群体定义_上访'] = ['两会', '二会', '访', '闹', '京', '维权']
##
SF_KEYWORD = OrderedDict()
SF_KEYWORD['上访事由_房地产'] = [
    '不动产证', '业主', '交付', '交房', '住房', '停工', '回迁', '土地使用权证', '土地所有权证', '土地权证',
    '土地证', '小区', '建设商', '开发企业', '开发商', '强制拆迁', '强制拆除', '强拆', '强行拆除', '房主',
    '房产证', '房地产', '房子', '房屋', '拆建', '拆房', '拆迁', '拆除', '搬迁回迁', '未完工', '未竣工',
    '烂尾'
]
SF_KEYWORD['上访事由_房地产_具体分类'] = {
    '交房困难': ['未完工', '未竣工', '烂尾', '未交', '停工', '拖延', '延期'],
    '拆迁相关': ['强制拆迁', '强制拆除', '强拆', '强行拆除', '拆房', '拆迁', '拆除', '回迁', '拆建'],
    '产权证': ['不动产证', '土地使用权证', '土地所有权证', '土地权证', '土地证', '房产证'],
    '开发商相关': ['建设商', '开发企业', '开发商'],
    '周边配套': [],
    '学区': ['学区'],
    '其他': []
}
##
SF_KEYWORD['上访事由_投资理财'] = [
    '商铺', '亏损', '保险统筹', '募集资金', '投资', '投资理财', '挽回损失', '损失', '筹资', '统保计划',
    '统筹保险', '综合保险', '资本投入', '资金亏损', '资金募集', '资金受损', '资金损失', '资金损毁', '资金运作',
    '追回赃款', '追缴赃款', '追赃挽损', '遭受损失', '集资'
]
SF_KEYWORD['上访事由_投资理财_具体分类'] = {
    '商铺相关': ['商铺'],
    '股票基金': ['股票', '基金'],
    '非法集资': ['p2p', 'P2P', 'p2P', 'P2p', '集资', '投资', '理财'],
    '合同纠纷': ['合同'],
    '品牌加盟': ['加盟'],
    '其他': []
}
##
SF_KEYWORD['上访事由_被诈骗'] = ['诈取', '诈骗', '诈骗行为', '欺诈', '欺骗', '骗取', '骗得']
##
SF_KEYWORD['上访事由_涉军'] = ['军人', '军官', '士兵', '战友', '涉军', '老兵', '退伍', '退役']
SF_KEYWORD['上访事由_教育培训'] = ['分校', '学员', '学生', '校区', '培训']
SF_KEYWORD['上访事由_涉维'] = ['新疆', '维族', '涉维']
SF_KEYWORD['上访事由_涉藏'] = ['西藏', '涉藏']
SF_KEYWORD['上访事由_新冠疫苗'] = ['疫苗', '新冠', '科兴']
##
SF_KEYWORD['上访地点_北京'] = [
    'BJ', 'beijing', '中办', '中南海', '中央办', '中央办公厅', '中央监察机关', '中央纪委',
    '中央纪律检查委员会', '中纪委', '京办', '京城', '北上', '北京', '升国旗', '国务院', '国家信访办', '国家信访局',
    '国家检察院', '天安门', '帝都', '最高人民检察院', '最高检', '检察机关', '赴京', '进京', '首都'
]
SF_KEYWORD['上访地点_省会'] = [
    '教育厅', '省事务', '省信访', '省公安厅', '省城', '省政府', '省级', '省部级', '赴省', '进省',
    '退役军人事务厅'
]
SF_KEYWORD['上访地点_市区'] = [
    '市信访', '市公安局', '市区', '市委', '市政府', '市教育局', '市级', '市级信访局', '退役军人事务局'
]
##
SF_KEYWORD['上访时间_两会'] = ['两会', '二会']
####
MODEL_PARA = OrderedDict()
MODEL_PARA['tf_house'] = Taskflow("zero_shot_text_classification",
                                  schema=list(
                                      SF_KEYWORD['上访事由_房地产_具体分类'].keys()))
MODEL_PARA['tf_invest'] = Taskflow("zero_shot_text_classification",
                                   schema=list(
                                       SF_KEYWORD['上访事由_投资理财_具体分类'].keys()))
MODEL_PARA['tf_pos_tagging'] = Taskflow('pos_tagging')


####
class ArmyKnife():
    """docstring for Help"""
    @staticmethod
    def format_change(res_list, cate, length=4):
        tmp = [
            re.sub(u"([^\u4e00-\u9fa5])", "", k[0]).strip() for k in res_list
            if k[1] == cate
            if len(re.sub(u"([^\u4e00-\u9fa5])", "", k[0])) >= length
        ]
        return tmp

    @staticmethod
    def key_word_match(keywords, text):
        if len(keywords) == 0:
            return dict()
        pattern = re.compile(r'|'.join(keywords))
        matches = pattern.findall(text)
        return dict(Counter(matches))

    @staticmethod
    def person_role_judge(text):
        ####
        leader_pattern = re.compile(r'|'.join(KEYWORD['人员角色_煽动者']))
        leader_matches = leader_pattern.findall(text)
        ####
        worker_pattern = re.compile(r'|'.join(KEYWORD['人员角色_响应者']))
        worker_matches = worker_pattern.findall(text)
        ####
        if leader_matches:
            return 'leader'
        elif worker_matches:
            return 'worker'
        else:
            return ''

    @staticmethod
    def find_closest(nums, value):
        # 使用 bisect 来找到插入位置
        pos = bisect.bisect_left(nums, value)
        # 如果插入位置在数组的最左边，则返回第一个元素
        if pos == 0:
            return nums[0]
        # 如果插入位置在数组的最右边，则返回最后一个元素
        if pos == len(nums):
            return nums[-1]
        # 比较插入位置附近的两个元素，选择更接近 value 的一个
        before = nums[pos - 1]
        after = nums[pos]
        # check = min(after - value, value - before)
        # if check >= 5:
        #     return -1
        if after - value < value - before:
            return after
        else:
            return before


####
class PosTaggingMixin:
    def pos_tagging_info(self):
        self.content_pos_tagging = MODEL_PARA['tf_pos_tagging'](self.content)
        self.get_time()
        self.get_name()
        self.get_organization()
        if self.title:
            self.title_pos_tagging = MODEL_PARA['tf_pos_tagging'](self.title)
            self.get_leader_pos_tagging()
            self.get_community()


####
class BasicInfo(PosTaggingMixin):
    # 其他基本信息提取方法
    def __init__(self, title, content):
        self.title = title  #re.sub(' +', ',', title)
        self.content = '【标题】' + '\n' * 2 + title + '\n' * 2 + ' 【内容】' + '\n' * 2 + content
        self.res = defaultdict(list)

    def re_info(self):
        ## 抽各实体元素
        self.get_idcard()
        self.get_pn()
        self.get_vx_id_name()
        self.get_vx_id()
        self.get_vx_nickname()
        self.get_huji_address()
        self.get_vx_group_name()
        self.get_vx_group_id()
        self.get_vx_group_members()
        self.get_vx_group_creator_id()

    def get_idcard(self):
        self.res['身份证号'] = jionlp.extract_id_card(self.content, detail=False)
        return self.res['身份证号']

    def get_pn(self):
        self.res['手机号码'] = jionlp.extract_phone_number(self.content,
                                                       detail=False)

    def get_vx_id_name(self):
        ## 从jionlp里面抽取
        tmp1 = jionlp.extract_wechat_id(self.content, detail=False)
        ## 正则模式1
        vx_id_name_pattern = r'微信号[:：]\s*(wxid_[a-zA-Z0-9]+)|【微信号】\s*\d*\s*【\s*(wxid_[a-zA-Z0-9]+)?】|【微信号】([a-zA-Z0-9_]+)|微信号：([a-zA-Z0-9]+)'
        vx_id_name = re.findall(vx_id_name_pattern, self.content)
        tmp2 = [k[0] for k in vx_id_name]
        ## 正则模式2
        tmp3 = re.findall(r'wxid_[a-zA-Z0-9]+', self.content)
        ##
        self.res['微信号'] = tmp1 + tmp2 + tmp3

    def get_vx_id(self):
        vx_id_pattern = r'(创建人ID|微信ID|微信用户ID)[:：](\d+)'
        # 提取微信用户ID
        vx_ids = re.findall(vx_id_pattern, self.content)
        self.res['微信ID'] = [k[1] for k in vx_ids]

    def get_huji_address(self):
        # 正则表达式模式
        pattern = r'地址[:：]\s*([^，；。\n]+)|户籍地址[:：]\s*([^，；。\n]+)|户籍地[:：]\s*([^，；。\n]+)'
        ##pattern = r'(?:地址|户籍(?:地址|地))[:：]\s*([^，。\n]+)'
        # 提取地址
        matches = re.findall(pattern, self.content)
        # 展平并过滤结果
        addresses = [
            address for match in matches for address in match if address
        ]
        self.res['地址'] = addresses

    def get_vx_nickname(self):
        # 正则表达式模式
        pattern = r'昵称[:：]\s*([^\)，）。\n]+)'
        # 提取地址
        nicknames = re.findall(pattern, self.content)
        # 展平并过滤结果
        self.res['昵称'] = nicknames

    def get_vx_group_name(self):
        group_name_pattern = r'(?:在\s*|微信群\s*|内\s*)[“"【]([^"”】]+)[”"】]|“([^"”“”]+)”微信群'
        # 提取微信群名称
        group_names = re.findall(group_name_pattern, self.content)
        self.res['微信群名称'] = [
            k[0].strip() for k in group_names if len(k[0].strip()) >= 2
        ]

    def get_vx_group_id(self):
        group_id_pattern = r'[（\(【]?(?:群ID|群id|群号|群号码)[:：]?(\d+)'  # 提取群ID，包括方括号形式
        group_ids = re.findall(group_id_pattern, self.content)
        self.res['微信群ID'] = group_ids

    def get_vx_group_members(self):
        group_members_pattern = r'(?:群成员数量|群成员数?|群人数)[:：]?\s?(\d+)'  # 提取群成员数量，包括方括号形式
        group_members = re.findall(group_members_pattern, self.content)
        self.res['微信群人数'] = group_members

    def get_vx_group_creator_id(self):
        # 使用正则表达式提取创建人ID
        pattern = r'创建人ID[:：]?\s*(\d+)[）)]?|创建人[:：]?\s*(\d+)'
        self.res['微信群创建人ID'] = [
            m[0] or m[1] for m in re.findall(pattern, self.content)
        ]

    def get_leader_pos_tagging(self):
        tmp1 = ArmyKnife.format_change(self.title_pos_tagging, 'PER',
                                       2)  ## 仅保留中文
        # print(tmp1)
        self.res['煽动者'].extend(tmp1)

    def get_community(self):
        tmp1 = ArmyKnife.format_change(self.title_pos_tagging, 'LOC', 4)
        self.res['小区名称'] = [k for k in tmp1 if not re.findall(r'省|市|县', k)]

    def get_name(self):
        ##
        self.res['姓名'] = ArmyKnife.format_change(self.content_pos_tagging,
                                                 'PER', 2)

    def get_time(self):
        ##
        self.res['时间'] = [
            k[0].strip() for k in self.content_pos_tagging if k[1] == 'TIME'
            if len(k[0].strip()) >= 3
        ]

    def get_organization(self):
        ##
        self.res['机构名称'] = ArmyKnife.format_change(self.content_pos_tagging,
                                                   'ORG', 4)  ## 是否需要进一步拆开

    # 嫌疑设备
    def get_imsi(self):
        pass

    def get_imei(self):
        imei_list = []
        pattern = r'imei.*?([A-Za-z0-9]{15,17})'
        matches = re.finditer(pattern, self.content)
        for match in matches:
            imei_followed_id = match.group(1)
            imei_list.append(imei_followed_id)
        self.res['imei'] = imei_list

    def get_mac(self):
        pass

    # 涉案物品
    def get_item(self):
        pass

    def get_item_type():
        pass

    # 涉案单位
    def get_workplace(self):
        pass

    def get_taxpayer_id(self):
        pass

    def get_workplace_type(self):
        pass


####
class InfoRelation(BasicInfo):
    """docstring for InfoRelation"""
    def __init__(self, title, content):
        super().__init__(title, content)

    def person_role_judge(self, text):
        ####
        leader_pattern = re.compile(r'|'.join(KEYWORD['人员角色_煽动者']))
        leader_matches = leader_pattern.findall(text)
        ####
        worker_pattern = re.compile(r'|'.join(KEYWORD['人员角色_响应者']))
        worker_matches = worker_pattern.findall(text)
        ####
        if leader_matches:
            return '煽动者'
        elif worker_matches:
            return '响应者'
        else:
            return ''

    def get_person_relation(self):
        self.res['人员信息'] = []
        ####
        sentences_1 = self.content.split('\n')  ## 换行模式
        sentences_2 = re.split(r'(?=\b\d{10}\b)', self.content)  ## 微信ID的分割模式
        sentences_3 = re.split(r'(?=\b\d{11}\b)', self.content)  ## pn分割模式
        sentences_2 = [
            sentences_2[i][10:] + sentences_2[i + 1][:10]
            for i in range(len(sentences_2) - 1)
        ] if len(sentences_2) >= 2 else []
        sentences_3 = [
            sentences_3[i][11:] + sentences_3[i + 1][:11]
            for i in range(len(sentences_3) - 1)
        ] if len(sentences_3) >= 2 else []
        sentences = sentences_1 + [
            segment.strip() for segment in sentences_2 if segment.strip()
        ] + [segment.strip() for segment in sentences_3 if segment.strip()]
        ##
        for sentence in sentences:
            single_sentence = BasicInfo('', sentence)
            if len(single_sentence.get_idcard()) != 1:
                continue
            ##
            single_sentence.re_info()
            single_sentence.pos_tagging_info()
            single_info = single_sentence.res
            ##
            a = single_info['身份证号'][0]
            b = single_info['手机号码'][-1] if single_info['手机号码'] else ''
            c = single_info['姓名'][-1] if single_info['姓名'] else ''
            d = single_info['地址'][-1] if single_info['地址'] else ''
            e = single_info['微信ID'][-1] if single_info['微信ID'] else ''
            z = self.person_role_judge(sentence)
            if b + c + d + e + z == '':
                continue
            if z != '':
                self.res[z].extend([c])
            ##
            tmp = {'身份证号': a, '手机号码': b, '姓名': c, '地址': d, '微信ID': e, '角色': z}
            self.res['人员信息'].append(tmp)
        ## 按照idcard去重
        check_idcard = set()
        self.res['人员信息'] = [
            x for x in self.res['人员信息']
            if not (x['身份证号'] in check_idcard or check_idcard.add(x['身份证号']))
        ]
        ####
        # for k in self.res['personal_relation']:
        #     print(k)
        #     print('\n')


####
class AdvancedInfo(InfoRelation):
    """docstring for Basic"""
    def __init__(self, title, content):
        super().__init__(title, content)
        self.re_info()  ## 正则信息
        self.pos_tagging_info()  ## 基本模型信息
        self.get_person_relation()  ## 人员关系
        self.get_sf_group()  ## 群体标签、文本内容分类
        ####
        for k, v in self.res.items():
            if k == '人员信息':
                continue
            print(k, v)
            self.res[k] = "&&&".join(list(set(v)))
        ####
        self.res = json.dumps(self.res, ensure_ascii=False)

    def get_person_rule_by_position(self):
        entity_dicts = {'person': self.res['姓名'], 'leader': KEYWORD['leader']}
        lexicon_ner = jionlp.ner.LexiconNER(entity_dicts)
        result = lexicon_ner(self.content)
        ####
        person_dict = {
            dd['offset'][0]: dd['text']
            for dd in result if dd['type'] == 'person'
        }
        person_locations = sorted(person_dict.keys())
        ####
        leader_locations = [
            dd['offset'][0] for dd in result if dd['type'] == 'leader'
        ]
        if not leader_locations:
            return ''
        ####
        res_names = [
            person_dict[ArmyKnife.find_closest(person_locations, loc)]
            for loc in leader_locations
        ]
        return Counter(res_names).most_common(1)[0][0]

    ## 上访群体标签
    def get_sf_group_label(self):
        self.res['群体标签'] = set(['人员群体_上访群体'])
        for group_label in SF_KEYWORD:
            ## 过滤部分用于细化分类的 SF_KEYWORD
            if isinstance(SF_KEYWORD[group_label], dict):
                continue
            # print( group_label)
            ## 匹配到“上访事由”, 则跳过
            if ('上访事由'
                    in self.res['群体标签']) and group_label.startswith('上访事由_'):
                continue
            #### 关键字匹配
            check = ArmyKnife.key_word_match(SF_KEYWORD[group_label],
                                             self.content)
            ## 没有匹配到
            if (not check):
                continue
            ## 能够匹配到
            self.res['群体标签'].add(group_label)
            if group_label.split('_')[0] == '上访事由':
                self.res['群体标签'].add('上访事由')
        #### 如果没有找到线索的分类，就加一个其他
        if not any(k for k in self.res['群体标签'] if k.startswith('上访事由_')):
            self.res['群体标签'].add('上访事由_其他')
        #### 去除结果中的'cause'
        if '上访事由' in self.res['群体标签']:
            self.res['群体标签'].remove('上访事由')

    ## 具体细化的文本分类
    def get_sf_content_classification(self):
        self.res['上访事由_具体分类'] = set()
        ##
        for group_label in SF_KEYWORD:
            #### 过滤部分KEYWORD
            if not isinstance(SF_KEYWORD[group_label], dict):
                continue
            if '_'.join(group_label.split('_')[:2]) not in self.res['群体标签']:
                continue
            #### 正则匹配
            for key in SF_KEYWORD[group_label]:
                check = ArmyKnife.key_word_match(SF_KEYWORD[group_label][key],
                                                 self.content)
                if (not check):
                    continue
                self.res['上访事由_具体分类'].add(key)
            #### 模型匹配
            ## print(self.res['上访事由_具体分类'])
            content_cates = list(SF_KEYWORD[group_label].keys())
            if '上访事由_房地产' in self.res['群体标签']:
                content_cls = MODEL_PARA['tf_house'](self.content)
            elif '上访事由_投资理财' in self.res['群体标签']:
                content_cls = MODEL_PARA['tf_invest'](self.content)
            else:
                pass
            ####
            tmp = content_cls[0]['predictions']
            if tmp and (tmp[0]['label'] in self.res['上访事由_具体分类']):
                self.res['上访事由_具体分类'] = [tmp[0]['label']]
            else:
                self.res['上访事由_具体分类'] = self.res['上访事由_具体分类']
            ##
            if not self.res['上访事由_具体分类']:
                self.res['上访事由_具体分类'] = ['其他']

    def get_sf_group(self):
        if not ArmyKnife.key_word_match(KEYWORD['群体定义_上访'], self.content):
            self.res['人员群体'] = ['其他']
            ## 如果非上访群体，就不做后续的群体标签的细化
            return
        self.get_sf_group_label()
        self.get_sf_content_classification()
        ####
        for xx in self.res['群体标签']:
            self.res[xx.split('_')[0]].append(xx.split('_')[1])
        del self.res['群体标签']
