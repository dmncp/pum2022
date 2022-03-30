# Podstawy uczenia maszynowego

## Laboratorium 2 - PCA

## Damian Cyper, gr. pon. 12:50B



### Wykorzystane technologie

* Python 3.9

* numpy

* Pillow

* scikit-learn

  

# todo: dopisać 





### Przygotowanie zbioru danych

Do realizacji ćwiczenia wybrałem zestaw trzech rodzajów sztućców - łyżki, widelce i noże (po 20 zdjęć każdego rodzaju). Następnie każde zdjęcie przeskalowałem do rozmiaru 100x100 pikseli oraz ustawiłem skalę szarości. W tym celu stworzyłem dwie funkcje:

```python
def resize_image(img_path):
    # Create an Image Object from an Image
    img = Image.open(img_path)

    # Make the new 100x100 image
    resized_img = img.resize((100, 100))

    # Save the cropped image
    resized_img.save(img_path)
```

```python
def convert_to_grayscale(img_path):
    # Create an Image Object from an Image
    img = Image.open(img_path)

    # convert to grayscale and save
    gray_image = ImageOps.grayscale(img)
    gray_image.save(img_path)
```

Kolejnym krokiem była zamiana macierzy pikseli na wektory. W tym celu skorzystałem z numpy'a:

```python
def convert_matrix_to_vector(img_path):
    img = Image.open(img_path)
    matrix = np.asarray(img)
    return matrix.flatten()


# for each image convert pixel matrix to vector and append to list
def get_pixel_vectors():
    for dirname in os.listdir(dataset_path):
        for filename in os.listdir(dataset_path + dirname):
            path = dataset_path + dirname + '/' + filename
            datasets_vectors.append(convert_matrix_to_vector(path))
```



### Wycentrowanie zbioru danych

Gdy dane zostały już odpowiednio przygotowane, uzyskaliśmy listę 60 wektorów o 10 000 współrzędnych (10000D). W kolejnym kroku obliczyłem wartość średnią z każdej kolumny - idąc po każdej współrzędnej od 0 do 10000 sumujemy współrzędne z każdego wektora i dzielimy przez 60. W praktyce korzystamy z numpy'a:

```python
# calculate the mean values of each column.
mean_vector = np.mean(datasets_vectors, axis=0, dtype=int)
```

Następnie centrujemy wartości w każdej kolumnie przez odjęcie od listy wektorów wektora średnich, tzn. od współrzędnej 0 każdego wektora z listy odejmujemy pierwszą wartość z mean_vector, dla współrzędnych 1 odejmujemy drugą wartość itd.

```python
# Next, we need to center the values in each column by subtracting the mean column value.
centered = np.subtract(datasets_vectors, mean_vector)
```



### Jak wyglądało średnie zdjęcie (to które odjęliśmy od pozostałych, by wycentrować zbiór)?

Żeby to sprawdzić należy najpierw stworzyć macierz z wektora średnich. Następnie możemy skorzystać z metody .fromarray z numpy'a:

```python
def convert_vector_to_matrix(vector):
    matrix = np.reshape(vector, (100, 100))
    matrix = np.array(matrix, dtype=np.uint8)
    img = Image.fromarray(matrix)
    img.save('./mean_img.png')
```

Po zapisaniu zdjęcia wygląda ono następująco:

<img src="C:\Users\Damian Cyper\Desktop\STUDIA\Semestr 6\Uczenie maszynowe\laby\lab2-pca\pum2\mean_img.png" alt="mean_img" style="zoom:200%;" />