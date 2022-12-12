from aligndraw import AlignDraw # basemodels
from clipdraw import AlignDrawClip # basemodels
from clipdrawlan import AlignDrawClipLanv1 # basemodels
from clipdrawlan_v2 import AlignDrawClipLanv2 # basemodels



BUILDER = {
    'AlignDraw': AlignDraw,
    'AlignDrawClip': AlignDrawClip, # add
    'AlignDrawClipLan_v1': AlignDrawClipLanv1, # cat
    'AlignDrawClipLan_v2': AlignDrawClipLanv2 # cat
}