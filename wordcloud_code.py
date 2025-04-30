from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = "This is a sample text for generating a word cloud. Word clouds are a great way to visualize text data. The size of each word indicates its frequency or importance in the text. You can customize the appearance of the word cloud with different colors, shapes, and fonts."

wordcloud = WordCloud().generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()