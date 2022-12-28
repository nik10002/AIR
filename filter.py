import json
import os
import wget
import gzip
import html

def downloadData(url):
    return wget.download(url)


def loadJson(filename):
    data = []
    with gzip.open(filename, "rt") as f:
        for line in f:
            stripped = line.strip()
            stripped = stripped.replace("\'", "\"")
            try:
                data.append(json.loads(stripped))
            except json.decoder.JSONDecodeError:
                continue 
    return data


def filterJson(data, extrema_dict):
    filtered = []
    for object in data:
        if not isGoodJsonEntry(object):
            continue
        
        sales_rank_key = list(object['salesRank'].keys())[0]
        sales_rank_value = object['salesRank'][sales_rank_key]
        min = extrema_dict[sales_rank_key][0]
        max = extrema_dict[sales_rank_key][1]

        tmp = {}
        tmp['description'] = html.unescape(object['description'])
        tmp['salesRank'] = (sales_rank_value - min) / (max - min)
        filtered.append(tmp)

    return filtered


def getExtremaDict(data):
    extrema_dict = {}
    for object in data:
        if not isGoodJsonEntry(object):
            continue

        sales_rank_key = list(object['salesRank'].keys())[0]
        sales_rank_value = object['salesRank'][sales_rank_key]
        if sales_rank_key not in extrema_dict:
            extrema_dict[sales_rank_key] = [sales_rank_value, sales_rank_value] 
        else:
            if extrema_dict[sales_rank_key][0] > sales_rank_value:
                extrema_dict[sales_rank_key][0] = sales_rank_value
            elif extrema_dict[sales_rank_key][1] < sales_rank_value:
                extrema_dict[sales_rank_key][1] = sales_rank_value

    return extrema_dict


def isGoodJsonEntry(object):
    if 'description' not in object.keys():
        return False
    if len(object['description']) == 0:
        return False
    if 'salesRank' not in object.keys():
        return False
    if len(object['salesRank']) == 0:
        return False
    return True


def safeJson(filtered, filename):
    with open(filename, 'w') as f:
        for object in filtered:
            json.dump(object, f)
            f.write(os.linesep)


def main():
    #url = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz"
    #filename = downloadData(url)
    filename = "meta_Electronics.json.gz"
    data = loadJson(filename)
    extrema_dict = getExtremaDict(data)
    filtered = filterJson(data, extrema_dict)
    filename = "filtered_" + os.path.splitext(filename)[0]
    safeJson(filtered, filename)

if __name__ == "__main__":
    main()