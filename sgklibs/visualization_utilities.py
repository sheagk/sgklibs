#!/usr/bin/env python3


def pdfimage_convert(fname,outname):
    """
    use pdfimages (part of brew install poppler) to pull out the raw image
    
    note that this will leave off any overlays, such as scale bars in the PFH images,
    but it will also automatically trim the image.
    """
    import tempfile
    from subprocess import call
    file,temp = tempfile.mkstemp()
    tocall = ['pdfimages',fname,temp]
    e = call(tocall)
    assert e == 0
    fname = temp+'-000.ppm'
    tocall = ['convert',fname,outname]
    assert call(tocall) == 0
    return outname

def pdf_to_png(fname, outname=None, img_only=True, read_ops={'-density':600}, write_ops={'-quality':100}):
    """
    use ImageMagick convert to convert PDF to PNG.  if img_only,
    then it'll use poppler to pull out the image, so any scale bars
    etc. probably won't be transferred.
    """
    if outname is None:
        import tempfile
        file, outname = tempfile.mkstemp(suffix='.png')

    if img_only:
        return pdfimage_convert(fname,outname)

    from subprocess import call
    tocall = ['convert']
    for key in read_ops:
        tocall.append(key)
        tocall.append(str(read_ops[key]))
    tocall.append(fname)
    for key in write_ops:
        tocall.append(key)
        tocall.append(str(write_ops[key]))
    tocall.append(outname)

    print(' '.join(tocall))
    e = call(tocall)
    assert(e==0)
    return outname


def add_scalebar(fname, outname, img_width, sbar_length, text, lower_left=[0.05, 0.075], lw=5, text_below=True, color='white', font='STIXGeneral', fontsize=50, dpi=600, offset=0.01, use_PIL=False, sbar_height=0.025):
    """
    add a scale bar to an exist image (that fills the entire frame).

    :param fname: string
        filename to add scale bar to

    :param outname: string
        output filename

    img_width: float : 
        size of the image in units
    sbar_length : float :
        size of the scale bar to add, in same units
    bottom_left: list-like, float:
        x,y position of the bottom left of the scale bar, in 0-1 units
    sbar_height:  float
        height of the scale bar, in 0-1 units
    """

    if use_PIL:
        from PIL import Image, ImageDraw, ImageFont
        im = Image.open(fname)

        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype(font, fontsize)

        w = im.width
        h = im.height

        lower_left = [lower_left[0]*w, lower_left[1]*h]

        x1 = lower_left[0] + (sbar_length/img_width)*w
        y1 = lower_left[1] + sbar_height*h

        draw.rectangle([tuple(lower_left), (x1, y1)], outline=color, fill=color)
        text_x = (x1 + lower_left[0])/2.0
        text_y = (y1 + lower_left[1])/2.0
        # if text_below:
        #     text_y = lower_left[0] - sbar_height*h/2.
        # else:
        #     text_y = y1 + fontsize*2
        if text_below:
            text = '\n'+text
        else:
            text = text + '\n'

        draw.text([text_x, text_y], text, font=font, fill=color)
        im.save(outname, 'PNG')

    else:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        from matplotlib.ticker import NullLocator
        import numpy as np

        fontdict = {'family':font, 'size':fontsize, 'color':color}
        img = mpimg.imread(fname)

        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())
        for spine in ['top', 'left', 'bottom', 'right']:
            plt.setp(ax.spines[spine], visible=False)
        plt.setp(ax.get_xaxis(), visible=False)
        plt.setp(ax.get_yaxis(), visible=False)

        h, w, dim = img.shape
        long_length = max([h, w])
        if w == long_length:
            h = h/w
            w = 1
        else:
            w = w/h
            h = 1

        im = ax.imshow(img, extent=[0, w, 0, h])

        x0, y0 = lower_left
        x1 = x0 + (sbar_length/img_width)

        ax.plot([x0, x1], [y0, y0], ls='-', lw=lw, color=color)
        text_x = (x1 + x0)/2.0
        if text_below:
            va = 'top'
            text_y = y0 - offset
        else:
            va = 'bottom'
            text_y = y0 + offset
        ax.text(text_x, text_y, text, color=color, fontdict=fontdict, ha='center', va=va)
        plt.savefig(outname, dpi=dpi, bbox_inches='tight', pad_inches=0)


def crop_image(fname, outname, tot_width, desired_width, tot_height=None, desired_height=None):
    '''
    crop an image located in fname and save the output in outname.

    tot_width and desired_width should be in the same units, but can be any units.  

    if tot_height is None, then the image must be square and tot_height is 
    assumed to be tot_width

    if desired_height is None, then it's assumed you want a square image out
    and it's taken to be desired_width

    '''

    from PIL import Image
    from math import ceil
    from .low_level_utils import backup_file

    try:
        img = Image.open(fname)
    except OSError:
        print("Converting to a png...")
        fname = pdf_to_png(fname)
        img = Image.open(fname)
    ow,oh = img.size

    width_frac = desired_width / tot_width
    if tot_height is None:
        assert oh == ow, "image is not square; must supply tot_height"
        tot_height = tot_width
    if desired_height is None:
        desired_height = desired_width

    height_frac = desired_height / tot_height

    dw = int(ceil(ow * width_frac))
    dh = int(ceil(oh * height_frac))

    wstart = int(ceil((ow-dw)/2))
    hstart = int(ceil((oh-dh)/2))

    img2 = img.crop((wstart,hstart,wstart+dw,hstart+dh))

    backup_file(outname)
    img2.save(outname)