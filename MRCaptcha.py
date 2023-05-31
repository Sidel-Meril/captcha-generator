import random, math
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageOps
import numpy as np, os
from scipy.spatial.distance import cdist
from datetime import datetime

class SegoeSymbol():
    def __init__(self, symbol,font_size: int = 75, border: int = 4) -> None:
        self.symb_str = symbol
        self.font_size = font_size
        self.border = border
        self.font = ImageFont.truetype('segoepr.ttf', size = font_size)
        self.box = self.font.getbbox(symbol)
        self.symb_img = Image.new('RGBA', (self.box[2]-self.box[0]+self.border*2, self.box[3]-self.box[1]+self.border*2), (0,0,0,0))

    def __call__(self, inv=False) -> Image.Image:
        self.symb_editor = ImageDraw.Draw(self.symb_img)
        self.symb_editor.text((self.border-self.box[0],self.border-self.box[1]), self.symb_str,
                              font=self.font, fill=(255, 255, 255, 255), stroke_width=self.border, stroke_fill=(0,0,0,255))
        self.symb_img, param = self._quad_tramsform()
        self.box=list(self.box)
        self.symb_img = self.trim(self.symb_img)
        return self.symb_img

    def trim(self,im):
        bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            return im.crop(bbox)

    def _quad_tramsform(self):
        w, h = self.symb_img.size
        orig_w, orig_h = w, h
        dx, dy = random.uniform(2,-20), random.uniform(2,-20)
        dispose = lambda d: (abs((int(random.uniform(-d,d)))),abs((int(random.uniform(-d,d)))))
        x, y = dispose(dx), dispose(dy)
        w += x[0]+x[1]
        h += y[0]+y[1]
        transform_base = Image.new('RGBA', (w+x[0]+x[1],h+y[0]+y[1]), (0,0,0,0))
        transform_base.paste(self.symb_img, (x[0], y[0]), self.symb_img)
        mesh = (
            x[0], y[0],
            -x[0], h+y[1],
            w+x[1], h - y[1],
            w - x[1], y[0]
        )
        return transform_base.transform((int(transform_base.size[0]),int(transform_base.size[0]))
                                        , data=mesh,method=Image.QUAD, resample=Image.BILINEAR), x

    def warp_image(self, image):
        r = 10  # individually warp a bunch of 10x10 tiles.
        mesh_x = int((image.size[0] / r) + 2)
        mesh_y = int((image.size[1] / r) + 2)

        # Set up some random values we'll use over and over...
        amplitude = random.uniform(1.8, 3.5)
        period = random.uniform(0.45, 0.60)
        offset = (
            random.uniform(0, math.pi * 2 / period),
            random.uniform(0, math.pi * 2 / period),
        )

        def _sine(x, y, a=amplitude, p=period, o=offset):
            """ Given a single point, warp it.  """
            return (
                math.sin((y + o[0]) * p) * a + x,
                math.sin((x + o[1]) * p) * a + y,
            )

        def _clamp(x, y):
            """ Don't warp things outside the bounds of the image. """
            return (
                max(0, min(image.size[0] - 1, x)),
                max(0, min(image.size[1] - 1, y)),
            )

        # Build a map of the corners of our r by r tiles, warping each one.
        warp = [
            [
                _clamp(*_sine(i * r, j * r))
                for j in range(mesh_y)
            ] for i in range(mesh_x)
        ]

        def _destination_rectangle(i, j):
            """ Return a happy tile from the original space. """
            return (i * r, j * r, (i + 1) * r, (j + 1) * r)

        def _source_quadrilateral(i, j):
            """ Return the set of warped corners for a given tile.

            Specified counter-clockwise as a tuple.
            """
            return (
                warp[i][j][0], warp[i][j][1],
                warp[i][j + 1][0], warp[i][j + 1][1],
                warp[i + 1][j + 1][0], warp[i + 1][j + 1][1],
                warp[i + 1][j][0], warp[i + 1][j][1],
            )

        # Finally, prepare our list of sources->destinations for PIL.
        mesh = [
            (
                _destination_rectangle(i, j),
                _source_quadrilateral(i, j),
            )
            for j in range(mesh_y - 1)
            for i in range(mesh_x - 1)
        ]
        # And do it.
        return image.transform(image.size, Image.MESH, mesh, Image.BILINEAR)

class LineImage():
    def __init__(self, thin: int = 5, color: tuple = (0,0,0,255), img_size:tuple = (500,200)):
        self.line_img = Image.new('RGBA', img_size, (0,0,0,0))
        self._thin = thin
        self._color = color

    def __call__(self, n_curves = 3, color = (), *args, **kwargs):
        self.line_editor = ImageDraw.Draw(self.line_img)
        x = np.arange(0, self.line_img.size[0], 1)
        y = None
        amplitude = random.uniform(self.line_img.size[0]/8, self.line_img.size[0]/4)
        line_axis = random.uniform(amplitude, self.line_img.size[1]-amplitude)
        for n in range(n_curves):
            period = random.uniform(self.line_img.size[0]/4, self.line_img.size[0]/2)
            wavelength = 2*np.pi/period
            # line_axis = 0
            phase = random.uniform(0, period)
            print(amplitude, wavelength, phase, line_axis)
            sine = lambda x: amplitude*math.sin(wavelength*x+phase)
            sine = np.vectorize(sine)
            if type(y) != np.ndarray: y = sine(x)
            else: y+=sine(x)
        resize_rate = self.line_img.size[1]/((y.max()-y.min()+line_axis))
        self._points = list(zip(list(x), list(resize_rate*y+line_axis)))

        self.line_editor.line(self._points, fill=self._color, width=self._thin, joint='curve')
        return self.line_img

class CaptchaImage():
    def __init__(self, **kwargs) -> None:
        self.h = 200
        self.w = 500
        self.border = (5,25)
        self.base = Image.new('RGB', (self.w, self.h), (255,255,255))
        self.subbase = Image.new('RGB', (self.w, self.h), (255,255,255))
        self.blank = Image.new('RGB', (self.w, self.h), (255,255,255))
        self.symb_masks = []

    def put_text(self, text: str = 'KU1CAx', **kwargs):
        tborder=4
        symbols = [SegoeSymbol(symbol, border=tborder, **kwargs)() for symbol in text]
        text_w, text_h, text_h_min = sum([symbol.size[0] for symbol in symbols]), \
                         max([symbol.size[1] for symbol in symbols]), \
                         max([symbol.size[1] for symbol in symbols]),
        assert self.w-2*self.border[0]>text_w, f'Text has too large width: {text_w}px > {self.w}px'
        assert self.h-2*self.border[1]>text_h, f'Text has too large height: {text_h}px > {self.h}px'
        dist_symbols = [0]
        [dist_symbols.append(self.minimum_distance(symbols[i-1],symbols[i])) for i in range(1,len(symbols))]
        # text_w -=sum(dist_symbols)

        x = random.randint(self.border[0], self.w-self.border[0]-text_w)
        top = random.randint(self.border[1], (self.h-self.border[1]-text_h)//2)
        self._top = top
        resize_rate_y = (self.h-top-self.border[1])/text_h
        self._text_h = int(text_h*resize_rate_y)
        resize_rate_x = (self.w-x)/text_w
        symbols = [symbol.resize((int(symbol.size[0]*resize_rate_x),
                                            int((symbol.size[1]+(text_h-symbol.size[1])*.5)*resize_rate_y)))
                   for symbol in symbols]
        dist_symbols = list(map(lambda x: int(x*resize_rate_x), dist_symbols))
        for symbol,symbol_str, idx in zip(symbols, list(text), range(len(symbols))):
            x -= dist_symbols[idx]
            print("".join(text),symbol_str, x)
            self.base.paste(symbol, (x,top), symbol)
            self.subbase.paste(ImageOps.invert(symbol.convert('RGB')), (x,top), symbol)
            blank = self.blank.copy()
            blank.paste(ImageOps.invert(symbol.convert('RGB')), (x,top), symbol)
            self.symb_masks.append(blank)
            x += symbol.size[0]+5
        return self.base

    def put_line(self):
        sine_generator = LineImage(thin=10, color=(255,255,255,255), img_size=(self.base.size[0], self._text_h))
        sine_generator(n_curves=random.randint(2,3))
        sine_generator(n_curves=random.randint(2,3))
        line = sine_generator.line_img
        self.base.paste(line, (0, self._top), line)
        return self.base

    def minimum_distance(self, symbolA, symbolB):
        A, B = np.array(symbolA.convert("1")), np.array(symbolB.convert("1"))
        A_px, B_px =  np.array(np.where(A==1)), np.array(np.where(B==1))
        B_px[1,:]+=symbolA.size[0]
        product = cdist(A_px.T, B_px.T, 'euclidean' )
        imin = np.where(product==product.min())[0][0], np.where(product==product.min())[1][0]
        print(product.min(), imin, product.shape, A_px.shape, B_px.shape, B_px.T[imin[1]][1], A_px.T[imin[0]][1])
        return B_px.T[imin[1]][1]-A_px.T[imin[0]][1]

    def warp_image(self):
        image =  self.base
        r = 10  # individually warp a bunch of 10x10 tiles.
        mesh_x = int((image.size[0] / r) + 2)
        mesh_y = int((image.size[1] / r) + 2)

        # Set up some random values we'll use over and over...
        amplitude = random.uniform(1.8, 3.5)
        period = random.uniform(0.45, 0.60)
        offset = (
            random.uniform(0, math.pi * 2 / period),
            random.uniform(0, math.pi * 2 / period),
        )

        def _sine(x, y, a=amplitude, p=period, o=offset):
            """ Given a single point, warp it.  """
            return (
                math.sin((y + o[0]) * p) * a + x,
                math.sin((x + o[1]) * p) * a + y,
            )

        def _clamp(x, y):
            """ Don't warp things outside the bounds of the image. """
            return (
                max(0, min(image.size[0] - 1, x)),
                max(0, min(image.size[1] - 1, y)),
            )

        # Build a map of the corners of our r by r tiles, warping each one.
        warp = [
            [
                _clamp(*_sine(i * r, j * r))
                for j in range(mesh_y)
            ] for i in range(mesh_x)
        ]

        def _destination_rectangle(i, j):
            """ Return a happy tile from the original space. """
            return (i * r, j * r, (i + 1) * r, (j + 1) * r)

        def _source_quadrilateral(i, j):
            """ Return the set of warped corners for a given tile.

            Specified counter-clockwise as a tuple.
            """
            return (
                warp[i][j][0], warp[i][j][1],
                warp[i][j + 1][0], warp[i][j + 1][1],
                warp[i + 1][j + 1][0], warp[i + 1][j + 1][1],
                warp[i + 1][j][0], warp[i + 1][j][1],
            )

        # Finally, prepare our list of sources->destinations for PIL.
        mesh = [
            (
                _destination_rectangle(i, j),
                _source_quadrilateral(i, j),
            )
            for j in range(mesh_y - 1)
            for i in range(mesh_x - 1)
        ]
        # And do it.
        self.base = self.base.transform(image.size, Image.MESH, mesh, Image.BILINEAR)
        self.subbase = self.subbase.transform(image.size, Image.MESH, mesh, Image.BILINEAR)
        for i in range(len(self.symb_masks)):
            self.symb_masks[i]=self.symb_masks[i].transform(image.size, Image.MESH, mesh, Image.BILINEAR)
        return image.transform(image.size, Image.MESH, mesh, Image.BILINEAR)

    def crop_white(self,im):
        bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            return im.crop(bbox), bbox

class TextGenerator():
    def __init__(self, length: int =6):
        self.length = length
        self.symbols = list("123456789OERWTYQUIOPASDFGHJKLZXCVBNMqwertyuiopadfghjklzxcvbnm")
    def generate(self, count: int = 10):
        texts= []
        for i in range(count):
            random.shuffle(self.symbols)
            texts.append(self.symbols[:self.length])
        return texts

symbols = "123456789OERWTYQUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm"

def generate_symbols(count):
    for symbol in symbols*count:
        print(symbol)
        sb = SegoeSymbol(symbol)
        segoe_symb = sb()
        segoe_symb = ImageOps.invert(segoe_symb.convert('RGB'))
        segoe_symb =sb.warp_image(segoe_symb)
        segoy_symb = sb.trim(segoe_symb)
        name = datetime.now().strftime("%m%d%Y-%H%M%S")
        if not os.path.exists(f'symbols_validation/{symbol+"_"+str(symbol.islower())}/'): os.makedirs(f'symbols_validation/{symbol+"_"+str(symbol.islower())}/')
        segoe_symb.save(f'symbols_validation/{symbol+"_"+str(symbol.islower())}/{name}.png')

def generate_captcha(count):
    if not os.path.exists(f'./samples/separated/'): os.makedirs(f'./samples/separated/')
    if not os.path.exists(f'./samples/undistorted/'): os.makedirs(f'./samples/undistorted/')
    if not os.path.exists(f'./samples/distorted/'): os.makedirs(f'./samples/distorted/')
    if not os.path.exists(f'./samples/inverted/'): os.makedirs(f'./samples/inverted/')
    if not os.path.exists(f'./samples/uinverted/'): os.makedirs(f'./samples/uinverted/')
    if not os.path.exists(f'./samples/lined/'): os.makedirs(f'./samples/lined/')
    for text in TextGenerator(6).generate(count):
        if not os.path.exists(f'./samples/separated/{"".join(text)}/'):
            cptch = CaptchaImage()
            try:
                cptch.put_text(text)
            except Exception as e:
                print(e)
                continue
            cptch.base.save(f'./samples/undistorted/{"".join(text)}.png')
            cptch.subbase.save(f'./samples/uinverted/{"".join(text)}.png')
            cptch.base = cptch.warp_image()
            cptch.base.save(f'./samples/distorted/{"".join(text)}.png')
            cptch.subbase.save(f'./samples/inverted/{"".join(text)}.png')
            img = cptch.put_line()
            os.makedirs(f'./samples/separated/{"".join(text)}/')
            for symbol, i in zip(cptch.symb_masks, range(6)):
                symbol.save(f'./samples/separated/{"".join(text)}/{i}.png')
            img.save(f'./samples/lined/{"".join(text)}.png')

    for text in TextGenerator(6).generate(64*16):
        cptch = CaptchaImage()
        try:
            cptch.put_text(text)
        except:
            continue
        cptch.base.save(f'./samples/undistorted/{"".join(text)}.png')
        cptch.subbase.save(f'./samples/uinverted/{"".join(text)}.png')
        cptch.base = cptch.warp_image()
        cptch.base.save(f'./samples/distorted/{"".join(text)}.png')
        cptch.subbase.save(f'./samples/inverted/{"".join(text)}.png')
        img = cptch.put_line()
        img.save(f'./samples/lined/{"".join(text)}.png')

if __name__=='__main__':
    for text in TextGenerator(6).generate(3):
        cptch = CaptchaImage()
        try:
            cptch.put_text(text)
        except Exception as e:
            print(e)
            continue
        # cptch.base.save(f'./samples/undistorted/{"".join(text)}.png')
        cptch.warp_image()
        # cptch.base.save(f'./samples/distorted/{"".join(text)}.png')
        cptch.subbase.save(f'{"".join(text)}-sub.png')
        img = cptch.put_line()
        img.save(f'{"".join(text)}-lin.png')