# captcha-generator

This small project provides Mail.ru Captcha Generator with using NumPy and SciPy for model training purposes

![Original](#results/original/1.jfif)
![Generated](#results/generated/lined/1a8JK2.png)

# Usage

1. Clone repo

```bash
git clone -b main https://github.com/sidel-meril/mru-auto.git
```

2. Install NumPy and SciPy


```bash
python3.9 -m pip install numpy
```
```bash
python3.9 -m pip install scipy
```

3. Generate samples

+ Captcha generation

```Python
import MRCaptcha

# Count specifies number of generated captcha samples
generate_captcha(count=10)
```

*Note:* This method saves all images during generation steps. All samples is saved into separated folders. Final results is saved into /lined

+ Distorted symbols generation

```Python
import MRCaptcha

# Count specifies number of generated symbols for each from list
generate_symbols(count=10)
```

## License

[MIT](https://choosealicense.com/licenses/mit/)