
def Draw2DGraph(figure,x,y,color):
   figure.clear() # чистим график
   ax = figure.add_subplot() # надо
   scatter = ax.scatter(x,y,c = color,cmap='rainbow') # задаем точки
   ax.set_xlabel(x.name) # задаем название осей
   ax.set_ylabel(y.name)
   ax.legend(*scatter.legend_elements(), title="Classes")


def Draw3DGraph(figure,x,y,z,color):
    figure.clear() # чистим график
    ax = figure.add_subplot(projection='3d') # надо
    figure.tight_layout() # увеличиваем размер графика
    scatter = ax.scatter(x,y,z,c = color,cmap='rainbow') # задаем точки
    ax.set_xlabel(x.name) # задаем название осей
    ax.set_ylabel(y.name)
    ax.set_zlabel(z.name)
    ax.legend(*scatter.legend_elements(), title="Classes")

