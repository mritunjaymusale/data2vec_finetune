from __future__ import print_function, absolute_import, division
from collections import namedtuple
#--------------------------------------------------------------------------------
# Definitions from https://github.com/AutoNUE/public-code/blob/master/helpers/anue_labels.py
#--------------------------------------------------------------------------------

# a label and all meta information
IDDLabel = namedtuple( 'IDDLabel' , [
    'name'        , 
    'id'          ,
    'csId'        ,
    'csTrainId'   ,    
    'level4Id'    , 
    'level3Id'    , 
    'level2IdName', 
    'level2Id'    , 
    'level1Id'    , 
    'hasInstances', 
    'ignoreInEval', 
    'color'       , 
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------


IDDLabels = [
    #       name                     id    csId csTrainId  level4id   level3Id  category    level2Id  level1Id  hasInstances   ignoreInEval   color
    IDDLabel(  'road'                 ,  0   ,  7 ,     0 ,       0   ,     0  ,   'drivable'            , 0           , 0      , False        , False        , (128, 64,128)  ),
    IDDLabel(  'parking'              ,  1   ,  9 ,   255 ,       1   ,     1  ,   'drivable'            , 1           , 0      , False        , False         , (250,170,160)  ),
    IDDLabel(  'drivable fallback'    ,  2   ,  255,  255 ,       2   ,     1  ,   'drivable'            , 1           , 0      , False        , False         , ( 81,  0, 81)  ),
    IDDLabel(  'sidewalk'             ,  3   ,  8 ,     1 ,       3   ,     2  ,   'non-drivable'        , 2           , 1      , False        , False        , (244, 35,232)  ),
    IDDLabel(  'rail track'           ,  4   , 10 ,   255 ,       3   ,     3  ,   'non-drivable'        , 3           , 1      , False        , False         , (230,150,140)  ),
    IDDLabel(  'non-drivable fallback',  5   , 255 ,    9 ,      4   ,      3  ,   'non-drivable'        , 3           , 1      , False        , False        , (152,251,152)  ),
    IDDLabel(  'person'               ,  6   , 24 ,    11 ,       5   ,     4  ,   'living-thing'        , 4           , 2      , True         , False        , (220, 20, 60)  ),
    IDDLabel(  'animal'               ,  7   , 255 ,   255 ,      6   ,     4  ,   'living-thing'        , 4           , 2      , True         , True        , (246, 198, 145)),
    IDDLabel(  'rider'                ,  8   , 25 ,    12 ,       7   ,     5  ,   'living-thing'        , 5           , 2      , True         , False        , (255,  0,  0)  ),
    IDDLabel(  'motorcycle'           ,  9   , 32 ,    17 ,       8   ,     6  ,   '2-wheeler'           , 6           , 3      , True         , False        , (  0,  0,230)  ),
    IDDLabel(  'bicycle'              , 10   , 33 ,    18 ,       9   ,     7  ,   '2-wheeler'           , 6           , 3      , True         , False        , (119, 11, 32)  ),
    IDDLabel(  'autorickshaw'         , 11   , 255 ,   255 ,     10   ,     8  ,   'autorickshaw'        , 7           , 3      , True         , False        , (255, 204, 54) ),
    IDDLabel(  'car'                  , 12   , 26 ,    13 ,      11   ,     9  ,   'car'                 , 7           , 3      , True         , False        , (  0,  0,142)  ),
    IDDLabel(  'truck'                , 13   , 27 ,    14 ,      12   ,     10 ,   'large-vehicle'       , 8           , 3      , True         , False        , (  0,  0, 70)  ),
    IDDLabel(  'bus'                  , 14   , 28 ,    15 ,      13   ,     11 ,   'large-vehicle'       , 8           , 3      , True         , False        , (  0, 60,100)  ),
    IDDLabel(  'caravan'              , 15   , 29 ,   255 ,      14   ,     12 ,   'large-vehicle'       , 8           , 3      , True         , True         , (  0,  0, 90)  ),
    IDDLabel(  'trailer'              , 16   , 30 ,   255 ,      15   ,     12 ,   'large-vehicle'       , 8           , 3      , True         , True         , (  0,  0,110)  ),
    IDDLabel(  'train'                , 17   , 31 ,    16 ,      15   ,     12 ,   'large-vehicle'       , 8           , 3      , True         , True        , (  0, 80,100)  ),
    IDDLabel(  'vehicle fallback'     , 18   , 355 ,   255 ,     15   ,     12 ,   'large-vehicle'       , 8           , 3      , True         , False        , (136, 143, 153)),  
    IDDLabel(  'curb'                 , 19   ,255 ,   255 ,      16   ,     13 ,   'barrier'             , 9           , 4      , False        , False        , (220, 190, 40)),
    IDDLabel(  'wall'                 , 20   , 12 ,     3 ,      17   ,     14 ,   'barrier'             , 9           , 4      , False        , False        , (102,102,156)  ),
    IDDLabel(  'fence'                , 21   , 13 ,     4 ,      18   ,     15 ,   'barrier'             , 10           , 4      , False        , False        , (190,153,153)  ),
    IDDLabel(  'guard rail'           , 22   , 14 ,   255 ,      19   ,     16 ,   'barrier'             , 10          , 4      , False        , False         , (180,165,180)  ),
    IDDLabel(  'billboard'            , 23   , 255 ,   255 ,     20   ,     17 ,   'structures'          , 11           , 4      , False        , False        , (174, 64, 67) ),
    IDDLabel(  'traffic sign'         , 24   , 20 ,     7 ,      21   ,     18 ,   'structures'          , 11          , 4      , False        , False        , (220,220,  0)  ),
    IDDLabel(  'traffic light'        , 25   , 19 ,     6 ,      22   ,     19 ,   'structures'          , 11          , 4      , False        , False        , (250,170, 30)  ),
    IDDLabel(  'pole'                 , 26   , 17 ,     5 ,      23   ,     20 ,   'structures'          , 12          , 4      , False        , False        , (153,153,153)  ),
    IDDLabel(  'polegroup'            , 27   , 18 ,   255 ,      23   ,     20 ,   'structures'          , 12          , 4      , False        , False         , (153,153,153)  ),
    IDDLabel(  'obs-str-bar-fallback' , 28   , 255 ,   255 ,     24   ,     21 ,   'structures'          , 12          , 4      , False        , False        , (169, 187, 214) ),  
    IDDLabel(  'building'             , 29   , 11 ,     2 ,      25   ,     22 ,   'construction'        , 13          , 5      , False        , False        , ( 70, 70, 70)  ),
    IDDLabel(  'bridge'               , 30   , 15 ,   255 ,      26   ,     23 ,   'construction'        , 13          , 5      , False        , False         , (150,100,100)  ),
    IDDLabel(  'tunnel'               , 31   , 16 ,   255 ,      26   ,     23 ,   'construction'        , 13          , 5      , False        , False         , (150,120, 90)  ),
    IDDLabel(  'vegetation'           , 32   , 21 ,     8 ,      27   ,     24 ,   'vegetation'          , 14          , 5      , False        , False        , (107,142, 35)  ),
    IDDLabel(  'sky'                  , 33   , 23 ,    10 ,      28   ,     25 ,   'sky'                 , 15          , 6      , False        , False        , ( 70,130,180)  ),
    IDDLabel(  'fallback background'  , 34   , 255 ,   255 ,     29   ,     25 ,   'object fallback'     , 15          , 6      , False        , False        , (169, 187, 214)),
    IDDLabel(  'unlabeled'            , 35   ,  0  ,   255 ,     255  ,     255 ,  'void'                , 255         , 255    , False        , True         , (  0,  0,  0)  ),
    IDDLabel(  'ego vehicle'          , 36   ,  1  ,   255 ,     255  ,     255 ,  'void'                , 255         , 255    , False        , True         , (  0,  0,  0)  ),
    IDDLabel(  'rectification border' , 37   ,  2  ,   255 ,     255  ,     255 ,  'void'                , 255         , 255    , False        , True         , (  0,  0,  0)  ),
    IDDLabel(  'out of roi'           , 38   ,  3  ,   255 ,     255  ,     255 ,  'void'                , 255         , 255    , False        , True         , (  0,  0,  0)  ),
    IDDLabel(  'license plate'        , 39   , 255 ,   255 ,     255  ,     255 ,  'vehicle'             , 255         , 255    , False        , True         , (  0,  0,142)  ),
    
]           






    

################ cityscapes ###################
# from https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
#
# Cityscapes labels
#



#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
CityScapesLabel = namedtuple( 'CityScapesLabel' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )




CityScapesLabels = [
    #                   name                   id    trainId   category            catId     hasInstances   ignoreInEval   color
    CityScapesLabel(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    CityScapesLabel(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    CityScapesLabel(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    CityScapesLabel(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    CityScapesLabel(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    CityScapesLabel(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    CityScapesLabel(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    CityScapesLabel(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    CityScapesLabel(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    CityScapesLabel(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    CityScapesLabel(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    CityScapesLabel(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    CityScapesLabel(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    CityScapesLabel(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    CityScapesLabel(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    CityScapesLabel(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    CityScapesLabel(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    CityScapesLabel(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    CityScapesLabel(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    CityScapesLabel(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    CityScapesLabel(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    CityScapesLabel(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    CityScapesLabel(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    CityScapesLabel(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    CityScapesLabel(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    CityScapesLabel(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    CityScapesLabel(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    CityScapesLabel(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    CityScapesLabel(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    CityScapesLabel(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    CityScapesLabel(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    CityScapesLabel(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    CityScapesLabel(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    CityScapesLabel(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    CityScapesLabel(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]



def getNamesandColors(Label, labels,dataset_type):

    # fixing the unlabed class and sorting the dictionary
    if dataset_type =='cityscapes':
        trainId2label   = { label.trainId : label for label in labels }
        trainId2label[255] = Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) )
    else:
        trainId2label   = { label.level3Id : label for label in labels }
        trainId2label[255]= Label(  'unlabeled'            , 35   ,  0  ,     255 ,   255   ,      255 ,   'void'                , 255         , 255    , False        , True         , (  0,  0,  0)  )
    trainId2label = dict(sorted(trainId2label.items()))

    class_names =[]
    class_colors =[]
    for key,value in trainId2label.items():
        if   dataset_type =='cityscapes':
            if value.trainId != -1 :
                class_names.append(value.name)
                class_colors.append(value.color)
        else:
            class_names.append(value.name)
            class_colors.append(value.color)
    return class_names,class_colors




