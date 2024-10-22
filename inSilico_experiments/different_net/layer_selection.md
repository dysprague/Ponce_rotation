## resnet50 and resnet50_linf_8
1. ('35', '.layer1.2.BatchNorm2dbn3'), [('inshape', (256, 57, 57)), ('outshape', (256, 57, 57))] -> 256, 64, 16, 4, 1
2. ('79', '.layer2.3.BatchNorm2dbn3'), [('inshape', (512, 29, 29)), ('outshape', (512, 29, 29))] -> 512, 128, 32, 8, 2
3. ('143', '.layer3.5.BatchNorm2dbn3'), [('inshape', (1024, 15, 15)), ('outshape', (1024, 15, 15))] -> 1024, 256, 64, 16, 4
4. ('177', '.layer4.2.BatchNorm2dbn3'), [('inshape', (2048, 8, 8)), ('outshape', (2048, 8, 8))] -> 2048, 512, 128, 32, 8

## corenet_s
1. ('6', '.V1.BatchNorm2dnorm2'), OrderedDict([('inshape', (64, 57, 57)), ('outshape', (64, 57, 57))]) -> 64, 16, 4, 1
2. ('30', '.V2.BatchNorm2dnorm3_1'), OrderedDict([('inshape', (128, 29, 29)), ('outshape', (128, 29, 29))]) -> 128, 32, 8, 2
3. ('74', '.V4.BatchNorm2dnorm3_3'), OrderedDict([('inshape', (256, 15, 15)), ('outshape', (256, 15, 15))]) -> 256, 64, 16, 4, 1
4. ('98', '.IT.BatchNorm2dnorm3_1'), OrderedDict([('inshape', (512, 8, 8)), ('outshape', (512, 8, 8))]) -> 512, 128, 32, 8, 2

## alexnet
1. ('4', '.features.Conv2d3'), OrderedDict([('inshape', (64, 27, 27)), ('outshape', (192, 27, 27))]) -> 192, 64, 16, 4, 1
2. ('7', '.features.Conv2d6'), OrderedDict([('inshape', (192, 13, 13)), ('outshape', (384, 13, 13))]) -> 384, 128, 32, 8, 2
3. ('9', '.features.Conv2d8'), OrderedDict([('inshape', (384, 13, 13)), ('outshape', (256, 13, 13))]) -> 256, 64, 16, 4, 1
4. ('11', '.features.Conv2d10'), OrderedDict([('inshape', (256, 13, 13)), ('outshape', (256, 13, 13))]) -> 256, 64, 16, 4, 1