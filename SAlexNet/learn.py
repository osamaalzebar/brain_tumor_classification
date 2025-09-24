import re

txt = '123-456-789   123.456.789'

pattern = re.compile(r'[0-9-.]+')

matches = pattern.finditer(txt)

print(matches)

for match in matches:
    print(match)