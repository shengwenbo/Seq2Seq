#coding=utf-8

import jieba

if __name__ == "__main__":
    with open("../corpus/data.train", encoding="utf-8") as fin, open("../corpus/wrong-right.txt", "w", encoding="utf-8") as fout:
        count = 0
        for line in fin.readlines():
            [_, _, content] = line.strip().split("\t", 2);
            if content.find("\t") >= 0:
                [origin, corrects] = content.split("\t", 1)
                origin = " ".join(list(origin))
                if corrects:
                    corrects = corrects.split("\t")
                    for c in corrects:
                        c = " ".join(list(c))
                        fout.write("%s\t%s\n" % (origin, c))
                        fout.write("%s\t%s\n" % (c, c))
                        count += 1
                        if count > 1000:
                            exit(1)
                else:
                    fout.write("%s\t%s\n" % (origin, origin))
