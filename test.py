import re

text = "这是一段很长的文本，其中包含imei：8658880448375001，后面可能还有其他数字，但我们要找的是imei后面的15到17位数字字母组合，比如这个imei：123ABC45678901234，但也可能只有15位，如imei：567DEF89012345678。"


# 定义一个函数来查找符合条件的数字字母组合
def find_imei_followed_by_id(text):
    # 正则表达式匹配imei后跟任意非贪婪字符，然后是15到17位的数字字母组合
    pattern = r'imei.*?([A-Za-z0-9]{15,17})'
    matches = re.finditer(pattern, text)

    # 遍历所有匹配项
    for match in matches:
        # match.group(1) 是我们想要的数字字母组合
        imei_followed_id = match.group(1)
        print(f"Found: {imei_followed_id} (after 'imei:')")


# 调用函数
find_imei_followed_by_id(text)
