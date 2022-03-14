lang_data = list(open(f"./data/dataset/Charades/Charades_sta_test.txt", 'r'))
search_list=[]
for item in lang_data:
    if 'turn' in item:
        search_list.append(item)
f=open(f"./data/dataset/Charades/Charades_sta_turn.txt", 'a')
f.writelines(search_list)
