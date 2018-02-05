# -*- coding: utf-8 -*-
import numpy as np
import cv2

def getColor(r,g,b):
    if r<=55 and g<=55 and b<=55:
        return 0
    elif r>=205 and g>=205 and b>=205:
        return 1
    else:
        if r<=45:
            if g<=45:
                if b<=45:
                    return 2
                elif 45<b<=80:
                    return 3
                elif 80<b<=115:
                    return 3
                elif 115<b<=150:
                    return 3
                elif 150<b<=185:
                    return 3
                elif 185<b<=220:
                    return 3
                else:
                    return 3
            elif 45<g<=80:
                if b<=45:
                    return 4
                elif 45<b<=80:
                    return 4
                elif 80<b<=115:
                    return 3
                elif 115<b<=150:
                    return 3
                elif 150<b<=185:
                    return 3
                elif 185<b<=220:
                    return 3
                else:
                    return 3
            elif 80<g<=115:
                if b<=45:
                    return 4
                elif 45<b<=80:
                    return 4
                elif 80<b<=115:
                    return 5
                elif 115<b<=150:
                    return 3
                elif 150<b<=185:
                    return 3
                elif 185<b<=220:
                    return 3
                else:
                    return 3
            elif 115<g<=150:
                if b<=45:
                    return 4
                elif 45<b<=80:
                    return 4
                elif 80<b<=115:
                    return 4
                elif 115<b<=150:
                    return 5
                elif 150<b<=185:
                    return 5
                elif 185<b<=220:
                    return 5
                else:
                    return 3
            elif 150<g<=185:
                if b<=45:
                    return 4
                elif 45<b<=80:
                    return 4
                elif 80<b<=115:
                    return 4
                elif 115<b<=150:
                    return 5
                elif 150<b<=185:
                    return 5
                elif 185<b<=220:
                    return 5
                else:
                    return 5
            elif 185<g<=220:
                if b<=45:
                    return 4
                elif 45<b<=80:
                    return 4
                elif 80<b<=115:
                    return 4
                elif 115<b<=150:
                    return 4
                elif 150<b<=185:
                    return 5
                elif 185<b<=220:
                    return 5
                else:
                    return 5
            else:
                if b<=45:
                    return 4
                elif 45<b<=80:
                    return 4
                elif 80<b<=115:
                    return 4
                elif 115<b<=150:
                    return 4
                elif 150<b<=185:
                    return 4
                elif 185<b<=220:
                    return 5
                else:
                    return 5
        elif 45<r<=80:
            if g<=45:
                if b<=45:
                    return 6
                elif 45<b<=80:
                    return 7
                elif 80<b<=115:
                    return 7
                elif 115<b<=150:
                    return 7
                elif 150<b<=185:
                    return 7
                elif 185<b<=220:
                    return 7
                else:
                    return 7
            elif 45<g<=80:
                if b<=45:
                    return 4
                elif 45<b<=80:
                    return 2
                elif 80<b<=115:
                    return 2
                elif 115<b<=150:
                    return 7
                elif 150<b<=185:
                    return 7
                elif 185<b<=220:
                    return 7
                else:
                    return 7
            elif 80<g<=115:
                if b<=45:
                    return 4
                elif 45<b<=80:
                    return 4
                elif 80<b<=115:
                    return 3
                elif 115<b<=150:
                    return 3
                elif 150<b<=185:
                    return 3
                elif 185<b<=220:
                    return 3
                else:
                    return 3
            elif 115<g<=150:
                if b<=45:
                    return 4
                elif 45<b<=80:
                    return 4
                elif 80<b<=115:
                    return 4
                elif 115<b<=150:
                    return 5
                elif 150<b<=185:
                    return 5
                elif 185<b<=220:
                    return 5
                else:
                    return 3
            elif 150<g<=185:
                if b<=45:
                    return 4
                elif 45<b<=80:
                    return 4
                elif 80<b<=115:
                    return 4
                elif 115<b<=150:
                    return 4
                elif 150<b<=185:
                    return 3
                elif 185<b<=220:
                    return 3
                else:
                    return 3
            elif 185<g<=220:
                if b<=45:
                    return 4
                elif 45<b<=80:
                    return 4
                elif 80<b<=115:
                    return 4
                elif 115<b<=150:
                    return 5
                elif 150<b<=185:
                    return 5
                elif 185<b<=220:
                    return 5
                else:
                    return 5
            else:
                if b<=45:
                    return 4
                elif 45<b<=80:
                    return 4
                elif 80<b<=115:
                    return 4
                elif 115<b<=150:
                    return 4
                elif 150<b<=185:
                    return 4
                elif 185<b<=220:
                    return 5
                else:
                    return 5
        elif 80<r<=115:
            if g<=45:
                if b<=45:
                    return 6
                elif 45<b<=80:
                    return 7
                elif 80<b<=115:
                    return 7
                elif 115<b<=150:
                    return 7
                elif 150<b<=185:
                    return 7
                elif 185<b<=220:
                    return 7
                else:
                    return 7
            elif 45<g<=80:
                if b<=45:
                    return 6
                elif 45<b<=80:
                    return 6
                elif 80<b<=115:
                    return 7
                elif 115<b<=150:
                    return 7
                elif 150<b<=185:
                    return 7
                elif 185<b<=220:
                    return 7
                else:
                    return 7
            elif 80<g<=115:
                if b<=45:
                    return 4
                elif 45<b<=80:
                    return 2
                elif 80<b<=115:
                    return 2
                elif 115<b<=150:
                    return 2
                elif 150<b<=185:
                    return 7
                elif 185<b<=220:
                    return 7
                else:
                    return 7
            elif 115<g<=150:
                if b<=45:
                    return 4
                elif 45<b<=80:
                    return 4
                elif 80<b<=115:
                    return 2
                elif 115<b<=150:
                    return 2
                elif 150<b<=185:
                    return 2
                elif 185<b<=220:
                    return 3
                else:
                    return 3
            elif 150<g<=185:
                if b<=45:
                    return 4
                elif 45<b<=80:
                    return 4
                elif 80<b<=115:
                    return 4
                elif 115<b<=150:
                    return 5
                elif 150<b<=185:
                    return 5
                elif 185<b<=220:
                    return 3
                else:
                    return 3
            elif 185<g<=220:
                if b<=45:
                    return 4
                elif 45<b<=80:
                    return 4
                elif 80<b<=115:
                    return 4
                elif 115<b<=150:
                    return 4
                elif 150<b<=185:
                    return 5
                elif 185<b<=220:
                    return 5
                else:
                    return 5
            else:
                if b<=45:
                    return 4
                elif 45<b<=80:
                    return 4
                elif 80<b<=115:
                    return 4
                elif 115<b<=150:
                    return 4
                elif 150<b<=185:
                    return 5
                elif 185<b<=220:
                    return 5
                else:
                    return 5
        elif 115<r<=150:
            if g<=45:
                if b<=45:
                    return 6
                elif 45<b<=80:
                    return 7
                elif 80<b<=115:
                    return 7
                elif 115<b<=150:
                    return 7
                elif 150<b<=185:
                    return 7
                elif 185<b<=220:
                    return 7
                else:
                    return 7
            elif 45<g<=80:
                if b<=45:
                    return 6
                elif 45<b<=80:
                    return 6
                elif 80<b<=115:
                    return 7
                elif 115<b<=150:
                    return 7
                elif 150<b<=185:
                    return 7
                elif 185<b<=220:
                    return 7
                else:
                    return 7
            elif 80<g<=115:
                if b<=45:
                    return 6
                elif 45<b<=80:
                    return 6
                elif 80<b<=115:
                    return 2
                elif 115<b<=150:
                    return 7
                elif 150<b<=185:
                    return 7
                elif 185<b<=220:
                    return 7
                else:
                    return 7
            elif 115<g<=150:
                if b<=45:
                    return 4
                elif 45<b<=80:
                    return 4
                elif 80<b<=115:
                    return 2
                elif 115<b<=150:
                    return 2
                elif 150<b<=185:
                    return 2
                elif 185<b<=220:
                    return 7
                else:
                    return 7
            elif 150<g<=185:
                if b<=45:
                    return 4
                elif 45<b<=80:
                    return 4
                elif 80<b<=115:
                    return 4
                elif 115<b<=150:
                    return 4
                elif 150<b<=185:
                    return 2
                elif 185<b<=220:
                    return 5
                else:
                    return 3
            elif 185<g<=220:
                if b<=45:
                    return 4
                elif 45<b<=80:
                    return 4
                elif 80<b<=115:
                    return 4
                elif 115<b<=150:
                    return 4
                elif 150<b<=185:
                    return 5
                elif 185<b<=220:
                    return 5
                else:
                    return 5
            else:
                if b<=45:
                    return 4
                elif 45<b<=80:
                    return 4
                elif 80<b<=115:
                    return 4
                elif 115<b<=150:
                    return 4
                elif 150<b<=185:
                    return 4
                elif 185<b<=220:
                    return 5
                else:
                    return 5
        elif 150<r<=185:
            if g<=45:
                if b<=45:
                    return 6
                elif 45<b<=80:
                    return 6
                elif 80<b<=115:
                    return 7
                elif 115<b<=150:
                    return 7
                elif 150<b<=185:
                    return 7
                elif 185<b<=220:
                    return 7
                else:
                    return 7
            elif 45<g<=80:
                if b<=45:
                    return 6
                elif 45<b<=80:
                    return 6
                elif 80<b<=115:
                    return 7
                elif 115<b<=150:
                    return 7
                elif 150<b<=185:
                    return 7
                elif 185<b<=220:
                    return 7
                else:
                    return 7
            elif 80<g<=115:
                if b<=45:
                    return 6
                elif 45<b<=80:
                    return 6
                elif 80<b<=115:
                    return 6
                elif 115<b<=150:
                    return 7
                elif 150<b<=185:
                    return 7
                elif 185<b<=220:
                    return 7
                else:
                    return 7
            elif 115<g<=150:
                if b<=45:
                    return 8
                elif 45<b<=80:
                    return 6
                elif 80<b<=115:
                    return 6
                elif 115<b<=150:
                    return 7
                elif 150<b<=185:
                    return 7
                elif 185<b<=220:
                    return 7
                else:
                    return 7
            elif 150<g<=185:
                if b<=45:
                    return 4
                elif 45<b<=80:
                    return 4
                elif 80<b<=115:
                    return 2
                elif 115<b<=150:
                    return 2
                elif 150<b<=185:
                    return 2
                elif 185<b<=220:
                    return 2
                else:
                    return 7
            elif 185<g<=220:
                if b<=45:
                    return 4
                elif 45<b<=80:
                    return 4
                elif 80<b<=115:
                    return 4
                elif 115<b<=150:
                    return 4
                elif 150<b<=185:
                    return 4
                elif 185<b<=220:
                    return 5
                else:
                    return 5
            else:
                if b<=45:
                    return 4
                elif 45<b<=80:
                    return 4
                elif 80<b<=115:
                    return 4
                elif 115<b<=150:
                    return 4
                elif 150<b<=185:
                    return 4
                elif 185<b<=220:
                    return 5
                else:
                    return 5
        elif 185<r<=220:
            if g<=45:
                if b<=45:
                    return 9
                elif 45<b<=80:
                    return 9
                elif 80<b<=115:
                    return 10
                elif 115<b<=150:
                    return 10
                elif 150<b<=185:
                    return 7
                elif 185<b<=220:
                    return 7
                else:
                    return 7
            elif 45<g<=80:
                if b<=45:
                    return 11
                elif 45<b<=80:
                    return 6
                elif 80<b<=115:
                    return 10
                elif 115<b<=150:
                    return 10
                elif 150<b<=185:
                    return 7
                elif 185<b<=220:
                    return 7
                else:
                    return 7
            elif 80<g<=115:
                if b<=45:
                    return 11
                elif 45<b<=80:
                    return 11
                elif 80<b<=115:
                    return 10
                elif 115<b<=150:
                    return 10
                elif 150<b<=185:
                    return 10
                elif 185<b<=220:
                    return 7
                else:
                    return 7
            elif 115<g<=150:
                if b<=45:
                    return 6
                elif 45<b<=80:
                    return 6
                elif 80<b<=115:
                    return 10
                elif 115<b<=150:
                    return 10
                elif 150<b<=185:
                    return 10
                elif 185<b<=220:
                    return 7
                else:
                    return 7
            elif 150<g<=185:
                if b<=45:
                    return 8
                elif 45<b<=80:
                    return 6
                elif 80<b<=115:
                    return 6
                elif 115<b<=150:
                    return 10
                elif 150<b<=185:
                    return 10
                elif 185<b<=220:
                    return 7
                else:
                    return 7
            elif 185<g<=220:
                if b<=45:
                    return 4
                elif 45<b<=80:
                    return 4
                elif 80<b<=115:
                    return 4
                elif 115<b<=150:
                    return 4
                elif 150<b<=185:
                    return 2
                elif 185<b<=220:
                    return 2
                else:
                    return 7
            else:
                if b<=45:
                    return 4
                elif 45<b<=80:
                    return 4
                elif 80<b<=115:
                    return 4
                elif 115<b<=150:
                    return 4
                elif 150<b<=185:
                    return 4
                elif 185<b<=220:
                    return 5
                else:
                    return 5
        else:
            if g<=45:
                if b<=45:
                    return 9
                elif 45<b<=80:
                    return 9
                elif 80<b<=115:
                    return 9
                elif 115<b<=150:
                    return 10
                elif 150<b<=185:
                    return 10
                elif 185<b<=220:
                    return 10
                else:
                    return 7
            elif 45<g<=80:
                if b<=45:
                    return 11
                elif 45<b<=80:
                    return 9
                elif 80<b<=115:
                    return 10
                elif 115<b<=150:
                    return 10
                elif 150<b<=185:
                    return 10
                elif 185<b<=220:
                    return 10
                else:
                    return 7
            elif 80<g<=115:
                if b<=45:
                    return 11
                elif 45<b<=80:
                    return 11
                elif 80<b<=115:
                    return 10
                elif 115<b<=150:
                    return 10
                elif 150<b<=185:
                    return 10
                elif 185<b<=220:
                    return 10
                else:
                    return 10
            elif 115<g<=150:
                if b<=45:
                    return 11
                elif 45<b<=80:
                    return 11
                elif 80<b<=115:
                    return 10
                elif 115<b<=150:
                    return 10
                elif 150<b<=185:
                    return 10
                elif 185<b<=220:
                    return 10
                else:
                    return 10
            elif 150<g<=185:
                if b<=45:
                    return 11
                elif 45<b<=80:
                    return 11
                elif 80<b<=115:
                    return 10
                elif 115<b<=150:
                    return 10
                elif 150<b<=185:
                    return 10
                elif 185<b<=220:
                    return 10
                else:
                    return 10
            elif 185<g<=220:
                if b<=45:
                    return 8
                elif 45<b<=80:
                    return 8
                elif 80<b<=115:
                    return 6
                elif 115<b<=150:
                    return 6
                elif 150<b<=185:
                    return 10
                elif 185<b<=220:
                    return 10
                else:
                    return 10
            else:
                if b<=45:
                    return 8
                elif 45<b<=80:
                    return 8
                elif 80<b<=115:
                    return 8
                elif 115<b<=150:
                    return 8
                elif 150<b<=185:
                    return 8
                elif 185<b<=220:
                    return 0
                else:
                    return 0


def createResponse(outputs, inputs):
    categories = ["Hat", "Hair", "Glove", "Sunglasses",
              "Upper-clothes", "Dress", "Coat", "Socks", "Pants",
             "Jumpsuits", "Scarf", "Skirt", "Face", "Left-arm",
             "Right-arm", "Left-leg", "Right-leg", "Left-shoe", "Right-shoe"]
    colors = ["black", "white", "gray", "blue",
             "green", "lightblue", "brown", "purple",
             "yellow", "red", "pink", "orange"]
    response = {}

    orig_w, orig_h = inputs.shape[0:2]
    response["inputs_shape"] = [orig_w, orig_h]

    inputs = cv2.resize(inputs, (512, 512))

    for i, cate in enumerate(np.unique(outputs, return_counts=True)[0]):
        if cate in [0, 2, 13, 14, 15, 16, 17] or np.unique(outputs, return_counts=True)[1][i] < 300:
            continue
        else:
            _ = {"colors":{}}
            index = np.where(outputs == cate)
            x = int(np.median(index[1]))
            y = int(np.median(index[0]))
            _["coords"] = [x, y]
            coords = [(x,y) for x,y in zip(index[1], index[0])]
            nb_color = np.zeros((12))
            for j in coords:
                r,g,b = inputs[j[1], j[0], :]
                num = getColor(r,g,b)
                nb_color[num] += 1
            sum_pixel = nb_color.sum()
            color_rank = nb_color.argsort()[::-1]
            for k in color_rank:
                prop = nb_color[k]/sum_pixel
                if prop > 0.3:
                    _["colors"][colors[k]] = prop
            response[categories[cate-1]] = _

    return response

