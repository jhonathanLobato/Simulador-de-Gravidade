
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ===============================
# PARÂMETROS FÍSICOS E INICIAIS
# ===============================
G = 6.67430e-11   # Constante gravitacional (m³/kg/s²)
M = 5.972e24      # Massa da Terra (kg)
m = 1000          # Massa do corpo de teste (kg)
origin = np.array([0.0, 0.0])  # Corpo central no centro

# Grade do espaço (20x20 pontos)
x = np.linspace(-6e7, 6e7, 20)
y = np.linspace(-6e7, 6e7, 20)
X, Y = np.meshgrid(x, y)

# ===============================
# FUNÇÃO PARA GERAR CAMPOS
# ===============================
def gerar_campo_gravitacional(X, Y):
    Fx = np.zeros(X.shape)  # Componente x do campo vetorial
    Fy = np.zeros(X.shape)  # Componente y do campo vetorial
    V = np.zeros(X.shape)   # Campo escalar (potencial)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            dx = origin[0] - X[i, j]
            dy = origin[1] - Y[i, j]
            r = np.sqrt(dx**2 + dy**2) + 1e3  # Distância
            F = G * M / r**2                  # Intensidade da força
            Fx[i, j] = F * dx / r
            Fy[i, j] = F * dy / r
            V[i, j] = -G * M / r
    return Fx, Fy, V

# Geração dos campos
Fx, Fy, V = gerar_campo_gravitacional(X, Y)

# ===============================
# CÁLCULO DO GRADIENTE E DERIVADA DIRECIONAL
# ===============================
gradV_y, gradV_x = np.gradient(V, y[1]-y[0], x[1]-x[0])

# Escolha do ponto e direção (45°)
ponto_x, ponto_y = 3e7, 3e7
theta = np.pi / 4
v_unit = np.array([np.cos(theta), np.sin(theta)])

# Cálculo do gradiente no ponto escolhido
i = (np.abs(x - ponto_x)).argmin()
j = (np.abs(y - ponto_y)).argmin()
grad = np.array([gradV_x[j, i], gradV_y[j, i]])

# Derivada direcional
derivada_direcional = np.dot(grad, v_unit)

# ===============================
# CONFIGURAÇÃO DA FIGURA
# ===============================
fig, ax = plt.subplots()
ax.set_xlim(-6e7, 6e7)
ax.set_ylim(-6e7, 6e7)
ax.set_title("Simulador de Campo Gravitacional e Derivadas Direcionais")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.grid(True)

# Campo vetorial (força gravitacional)
ax.quiver(X, Y, Fx, Fy, color='gray', alpha=0.5, label="Campo Vetorial (Força)")

# Campo escalar (potencial gravitacional)
ax.contourf(X, Y, V, levels=50, cmap='viridis', alpha=0.6)

# Gradiente do campo escalar
ax.quiver(X, Y, gradV_x, gradV_y, color='lime', alpha=0.5, label="Gradiente do Campo Escalar")

# Vetor da direção escolhida e gradiente no ponto
escala = 1e7
ax.quiver(ponto_x, ponto_y, v_unit[0]*escala, v_unit[1]*escala, 
          color='red', scale=1, scale_units='xy', label='Direção v̂ (vermelha)')
ax.quiver(ponto_x, ponto_y, grad[0]*escala, grad[1]*escala, 
          color='orange', scale=1, scale_units='xy', label='∇V (laranja)')

# Texto da derivada direcional
ax.text(-5.8e7, 5.5e7, f"Derivada direcional: {derivada_direcional:.2e} J/kg/m", 
        fontsize=10, color='black')

# ===============================
# SIMULAÇÃO DO CORPO EM MOVIMENTO
# ===============================
pos = np.array([6e7, 0], dtype=float)
vel = np.array([0, 2000], dtype=float)

trajetoria_x = []
trajetoria_y = []
ponto_plot, = ax.plot([], [], 'ro', label="Corpo em Movimento")
linha_plot, = ax.plot([], [], 'b-', lw=1)

def atualizar(frame):
    global pos, vel
    dx = origin[0] - pos[0]
    dy = origin[1] - pos[1]
    r = np.sqrt(dx**2 + dy**2) + 1e3
    F = G * M * m / r**2
    Fx_obj = F * dx / r
    Fy_obj = F * dy / r
    a = np.array([Fx_obj/m, Fy_obj/m])
    vel += a * 60
    pos += vel * 60
    trajetoria_x.append(pos[0])
    trajetoria_y.append(pos[1])
    ponto_plot.set_data([pos[0]], [pos[1]])
    linha_plot.set_data(trajetoria_x, trajetoria_y)
    return ponto_plot, linha_plot

# ===============================
# LEGENDA E EXECUÇÃO DA ANIMAÇÃO
# ===============================
ax.legend(loc='upper right')
ani = animation.FuncAnimation(fig, atualizar, frames=500, interval=30, blit=True)
plt.show()
