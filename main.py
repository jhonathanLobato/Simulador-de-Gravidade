import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ===============================
# PARÂMETROS FÍSICOS E INICIAIS
# ===============================
G = 6.67430e-11                 # Constante gravitacional
M = 5.972e24                    # Massa da Terra (kg)
m = 1000                        # Massa do corpo (kg)
origin = np.array([0.0, 0.0])   # Centro do campo gravitacional

# ===============================
# GRADE DE PONTOS NO PLANO
# ===============================
x = np.linspace(-6e7, 6e7, 20)
y = np.linspace(-6e7, 6e7, 20)
X, Y = np.meshgrid(x, y)

# ===============================
# CAMPO GRAVITACIONAL E POTENCIAL
# ===============================
# Função para calcular o campo gravitacional e potencial
def gerar_campo_gravitacional(X, Y):
    Fx = np.zeros(X.shape)
    Fy = np.zeros(X.shape)
    V = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            dx = origin[0] - X[i, j]
            dy = origin[1] - Y[i, j]
            r = np.sqrt(dx**2 + dy**2) + 1e3
            F = G * M / r**2
            Fx[i, j] = F * dx / r
            Fy[i, j] = F * dy / r
            V[i, j] = -G * M / r
    return Fx, Fy, V

# Calcula o campo e potencial
Fx, Fy, V = gerar_campo_gravitacional(X, Y)
# Gradiente do potencial
gradV_y, gradV_x = np.gradient(V, y[1]-y[0], x[1]-x[0])

# ===============================
# CONFIGURAÇÃO DA FIGURA
# ===============================
fig, ax = plt.subplots(figsize=(8,8)) # Tamanho da figura
ax.set_xlim(-6e7, 6e7) # Limites do gráfico
ax.set_ylim(-6e7, 6e7) # Limites do gráfico
ax.set_title("Simulador de Campo Gravitacional", fontsize=12)
ax.set_xlabel("x (m)") # Rótulo do eixo x
ax.set_ylabel("y (m)") # Rótulo do eixo y
ax.grid(True, alpha=0.3) # Grade de fundo

# Campo escalar (potencial) e streamlines
ax.contourf(X, Y, V, levels=50, cmap='plasma', alpha=0.6)
#ax.streamplot(X, Y, Fx, Fy, color='gray', density=1.2, linewidth=0.5, arrowsize=1)

# Terra no centro
terra = plt.Circle((0, 0), 6.371e6, color='blue', label="Terra")
ax.add_artist(terra)

# Textos dinâmicos
text_deriv = ax.text(-5.8e7, 5.5e7, "", fontsize=10, color='black')
text_grad = ax.text(-5.8e7, 5.0e7, "", fontsize=10, color='black')
text_vel = ax.text(-5.8e7, 4.5e7, "", fontsize=10, color='black')
text_força = ax.text(-5.8e7, 4.0e7, "", fontsize=10, color='black')

# Corpo em movimento
pos = np.array([6e7, 0], dtype=float) # Posição inicial
vel = np.array([0, 2000], dtype=float) # Velocidade inicial
trajetoria_x, trajetoria_y = [], [] # Trajetória
# Plots iniciais
ponto_plot, = ax.plot([], [], 'ro', markersize=6, label="Corpo")
linha_plot, = ax.plot([], [], '-', lw=1, color='blue', alpha=0.5)

# Vetores dinâmicos
escala = 1e7 # Escala para os vetores
quiver_vetor = ax.quiver([], [], [], [], color='red', scale=1, scale_units='xy')    # Vetor velocidade
quiver_grad = ax.quiver([], [], [], [], color='orange', scale=1, scale_units='xy')  # Vetor gradiente
quiver_força = ax.quiver([], [], [], [], color='green', scale=1, scale_units='xy')  # Vetor força


# ===============================
# Cria streamlines para mostrar o fluxo do campo
# ===============================
# PARTÍCULAS ANIMADAS PARA STREAMLINES
# ===============================
n_particulas = 100  # número de partículas
# Inicializa posições aleatórias das partículas na grade
particulas = np.zeros((n_particulas, 2))
particulas[:,0] = np.random.uniform(-6e7, 6e7, n_particulas)
particulas[:,1] = np.random.uniform(-6e7, 6e7, n_particulas)

# Plots das partículas
pontos_particulas, = ax.plot(particulas[:,0], particulas[:,1], 'yo', markersize=3, alpha=0.7)

# Função para atualizar partículas
def atualizar_particulas():
    global particulas
    # Para cada partícula, calcula o campo no ponto mais próximo da grade
    for k in range(n_particulas):
        i = (np.abs(x - particulas[k,0])).argmin()
        j = (np.abs(y - particulas[k,1])).argmin()
        # Campo vetorial no ponto da partícula
        vx = Fx[j,i]
        vy = Fy[j,i]
        # Normaliza para passo de movimento
        v = np.sqrt(vx**2 + vy**2)
        if v != 0:
            vx, vy = vx/v * 1e5, vy/v * 1e5  # ajusta velocidade visual
        particulas[k,0] += vx
        particulas[k,1] += vy
        # Se sair da tela, reinicia em posição aleatória
        if abs(particulas[k,0]) > 6e7 or abs(particulas[k,1]) > 6e7:
            particulas[k,0] = np.random.uniform(-6e7, 6e7)
            particulas[k,1] = np.random.uniform(-6e7, 6e7)
    # se atingir a terra, reinicia
        dx = particulas[k,0] - origin[0]
        dy = particulas[k,1] - origin[1]
        r = np.sqrt(dx**2 + dy**2)
        if r < 6.371e6:
            particulas[k,0] = np.random.uniform(-6e7, 6e7)
            particulas[k,1] = np.random.uniform(-6e7, 6e7)
    # Atualiza o plot das partículas
    pontos_particulas.set_data(particulas[:,0], particulas[:,1])


# ===============================
# FUNÇÃO DE ATUALIZAÇÃO
# ===============================
# Atualiza a posição, vetores e textos
# Função chamada a cada frame da animação
def atualizar(frame):
    global pos, vel, quiver_vetor, quiver_grad, quiver_força

    # Movimento
    dx = origin[0] - pos[0] # Distância x ao centro
    dy = origin[1] - pos[1] # Distância y ao centro
    r = np.sqrt(dx**2 + dy**2) + 1e3 # Distância total ao centro
    # Força gravitacional
    F = G * M * m / r**2
    Fx_obj = F * dx / r
    Fy_obj = F * dy / r
    # Atualiza velocidade e posição (passo de 60s)
    a = np.array([Fx_obj/m, Fy_obj/m])
    vel += a * 60
    pos += vel * 60

    # Atualiza trajetória
    trajetoria_x.append(pos[0])
    trajetoria_y.append(pos[1])
    ponto_plot.set_data([pos[0]], [pos[1]])
    linha_plot.set_data(trajetoria_x, trajetoria_y)
    linha_plot.set_alpha(0.3 + 0.7*frame/1000)  # efeito degradê

    # Gradiente no ponto atual
    i = (np.abs(x - pos[0])).argmin()
    j = (np.abs(y - pos[1])).argmin()
    grad = np.array([gradV_x[j, i], gradV_y[j, i]])

    # Vetor unitário da direção
    v_unit = vel / np.linalg.norm(vel) if np.linalg.norm(vel) > 0 else np.zeros(2)

    # Remove vetores antigos
    quiver_vetor.remove()
    quiver_grad.remove()
    quiver_força.remove()

    # Escala adaptativa
    escala_v = escala / np.linalg.norm(vel) if np.linalg.norm(vel) > 0 else escala
    escala_g = escala / np.linalg.norm(grad) if np.linalg.norm(grad) > 0 else escala

    # Atualiza vetores
    quiver_vetor = ax.quiver(pos[0], pos[1], v_unit[0]*escala_v, v_unit[1]*escala_v, color='red', scale=1, scale_units='xy')
    quiver_grad = ax.quiver(pos[0], pos[1], -grad[0]*escala_g, -grad[1]*escala_g, color='black', scale=1, scale_units='xy')
    quiver_força = ax.quiver(pos[0], pos[1], Fx_obj*escala/1e7, Fy_obj*escala/1e7, color='green', scale=1, scale_units='xy')

    # Atualiza textos
    derivada_direcional = np.dot(grad, v_unit)
    text_deriv.set_text(f"Derivada direcional: {derivada_direcional:.2e} J/kg/m")   # mostra a derivada direcional em J/kg/m
    text_grad.set_text(f"|∇V|: {np.linalg.norm(grad):.2e}")                         # mostra o gradiente em J/kg/m
    text_vel.set_text(f"|v|: {np.linalg.norm(vel):.2e} m/s ")                       # mostra a velocidade em m/s
    text_força.set_text(f"|F|: {np.linalg.norm([Fx_obj, Fy_obj]):.2e} N")           # mostra a força em N

    # Atualiza partículas animadas
    atualizar_particulas()

    # Retorna os elementos atualizados
    return ponto_plot, linha_plot, quiver_vetor, quiver_grad, quiver_força, text_deriv, text_grad, text_vel, text_força, pontos_particulas

# ===============================
# EXECUÇÃO DA ANIMAÇÃO
# ===============================
ax.legend(loc='upper right')
ani = animation.FuncAnimation(fig, atualizar, frames=1000, interval=20, blit=False)
plt.show()
