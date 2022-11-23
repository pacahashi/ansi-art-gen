from os import makedirs
from os.path import dirname, join

import cv2
import numpy as np
import scipy.spatial as sp


def main():
    SP = '\x20'
    D1 = '\xb0'
    D2 = '\xb1'
    D3 = '\xb2'
    LB = '\n'

    DIFF_D1 = 7.5
    DIFF_D2 = DIFF_D1*2
    DIFF_D3 = DIFF_D2*2

    zoom = 0.5
    columns = 80
    y_adjust = 0.5

    # 元画像の読み込み
    base_dir = dirname(__file__)
    dist_dir = join(base_dir, "dist")
    try:
        makedirs(dist_dir)
    except FileExistsError:
        pass

    # 読み込み
    img = cv2.imread(join(base_dir, "sample2.png"))

    # カラーパレットを調整
    b, g, r = cv2.split(img)
    b_min = np.amin(b)
    b_mid = np.median(b)
    b_max = np.amax(b)
    g_min = np.amin(g)
    g_mid = np.median(g)
    g_max = np.amax(g)
    r_min = np.amin(r)
    r_mid = np.median(r)
    r_max = np.amax(r)

    print(f'{b_min=}, {b_mid=}, {b_max=}\n{g_min=}, {g_mid=}, {g_max=}\n{r_min=}, {r_mid=}, {r_max=}')

    main_colors = [
        (b_min, g_min, r_min, '30', '40'),

        (b_min, g_min, r_max, '31', '41'),
        (b_min, g_max, r_min, '32', '42'),
        (b_min, g_max, r_max, '33', '43'),
        (b_max, g_min, r_min, '34', '44'),
        (b_max, g_min, r_max, '35', '45'),
        (b_max, g_max, r_min, '36', '46'),
        (b_max, g_max, r_max, '37', '47'),

        # (b_min, g_min, r_max, '1;31', '1;41'),
        # (b_min, g_max, r_min, '1;32', '1;42'),
        # (b_min, g_max, r_max, '1;33', '1;43'),
        # (b_max, g_min, r_min, '1;34', '1;44'),
        # (b_max, g_min, r_max, '1;35', '1;45'),
        # (b_max, g_max, r_min, '1;36', '1;46'),
        # (b_max, g_max, r_max, '1;37', '1;47'),

        # (b_min, g_min, r_mid, '31', '41'),
        # (b_min, g_mid, r_min, '32', '42'),
        # (b_min, g_mid, r_mid, '33', '43'),
        # (b_mid, g_min, r_min, '34', '44'),
        # (b_mid, g_min, r_mid, '35', '45'),
        # (b_mid, g_mid, r_min, '36', '47'),
        # (b_mid, g_mid, r_mid, '37', '47'),
    ]

    def sgr_color(n):
        return f'\033[{n}m'

    def ansi_code(
        color_main,
        color_sub,
        distance_main,
        distance_sub,
        color_current_main,
        color_current_sub,
    ):
        ret = ''

        # 背景色をメインカラーで描画
        if color_main != color_current_main:
            # 異なる場合はカラーチェンジする
            ret += sgr_color(main_colors[color_main][4])

        dis = abs(distance_main - distance_sub)

        if dis >= DIFF_D3:
            # 2色の差がとても大きいので1色に丸める
            # 背景色のみ表示のため空白文字
            ret += SP
        else:
            # 前景色をサブカラーで描画
            if color_sub != color_current_sub:
                # 異なる場合はカラーチェンジする
                ret += sgr_color(main_colors[color_main][3])

            # 2色の差がそこそこなので2色で描画
            if dis < DIFF_D1:
                # 2色の差が小さい
                ret += D1
            elif dis < DIFF_D2:
                # 2色の差がほどほど:
                ret += D2
            else:
                # 2色の差が大きい
                ret += D3

        return ret

    # 目標のドットサイズに変換する
    h, w, _ = img.shape
    img = cv2.resize(img, (int(w * zoom), int(h * zoom)))
    img = cv2.resize(img, (columns, int(columns*y_adjust)))

    # 変換後の各ドットに対して以下を決定する
    h, w, _ = img.shape
    chars = []
    current_main = -1
    current_sub = -1
    for py in range(0, h):
        for px in range(0, w):
            # 最も近い色と、次に近い色を抽出
            input_color = (img[py][px][0], img[py][px][1], img[py][px][2])
            tree = sp.KDTree([(b, g, r) for b, g, r, _, _ in main_colors])
            distance, result = tree.query(input_color, k=2)
            nearest_color = main_colors[result[0]]

            chars.append(ansi_code(
                color_main=result[0],
                color_sub=result[1],
                distance_main=distance[0],
                distance_sub=distance[1],
                color_current_main=current_main,
                color_current_sub=current_sub,
            ))
            current_main = result[0]
            current_sub = result[1]

            img[py][px][0] = nearest_color[0]
            img[py][px][1] = nearest_color[1]
            img[py][px][2] = nearest_color[2]

        # chars.append(LB)

    with open(join(dist_dir, 'out.ans'), 'wb') as f:
        f.write(''.join(chars).encode('latin-1'))

    cv2.imwrite(join(dist_dir, "out.png"), img)


main()
