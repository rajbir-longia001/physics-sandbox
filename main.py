import pygame
import pygame_gui
import json
import numpy as np

pygame.init()

# window setup
WIDTH, HEIGHT = 1000, 800
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Physics Sandbox')
GRID = pygame.image.load('transparent_grid.png')

manager = pygame_gui.UIManager((WIDTH, HEIGHT), theme_path='theme.json')

# fonts
TITLE_FONT = pygame.font.Font('bitcount.ttf', 75)
MAIN_FONT = pygame.font.Font('main_font.ttf', 18)

# sim constants
bg_color = 'black'
FPS = 60
clock = pygame.time.Clock()
g = 1000
friction = 0
rho = 0

class Ball:
    def __init__(self, radius, pos, vel, color):
        self.radius = radius
        self.x, self.y = pos
        self.xvel, self.yvel = vel
        self.color = color
        self.freefall = True
        self.checkpoint = None
        self.tangent_vector = None
        self.on_top = None

    def move(self, dt):
        if self.freefall:
            self.yvel += g * dt
        else:
            self.tangent_vector = self.tangent_vector / np.linalg.norm(self.tangent_vector)
            acc = self.tangent_vector * np.dot(self.tangent_vector, [0, g])
            vel = np.array([self.xvel, self.yvel])

            # if np.dot(vel, self.tangent_vector) > 0:
            #     self.xvel, self.yvel = self.tangent_vector * np.linalg.norm(vel)
            # elif np.dot(vel, self.tangent_vector) < 0:
            #     self.xvel, self.yvel = self.tangent_vector * -np.linalg.norm(vel)
            # else:
            #     self.xvel, self.yvel = 0, 0

            tangent_vel = self.tangent_vector * np.dot(self.tangent_vector, vel)
            tangent_vel += acc * dt
            
            self.xvel, self.yvel = tangent_vel

        self.x += self.xvel * dt
        self.y += self.yvel * dt
        


# help screen
def help(screen):
    run = True

    # UI elements
    help_screen = pygame_gui.elements.UIPanel(relative_rect=pygame.Rect(0, 0, WIDTH, HEIGHT), manager=manager)
    text_width, text_height = 700, 350
    help_text = pygame_gui.elements.UITextBox(html_text='''Thanks for checking out this physics sandbox. As of July 2025, there is one simulation under development, but I may add more later if time permits. To start the simulation, click the start button located in the menu. There are two modes: <font color="#ffff00">edit</font>, which can be enabled by pressing "Escape," allowing you to set initial conditions of the simulation; and <font color="#ffff00">run</font>, which can be enabled by pressing "Enter," running the simulation. Hopefully that helps. Have fun!''',
                                              relative_rect=pygame.Rect((WIDTH - text_width)/2, (HEIGHT - text_height)/2, text_width, text_height),
                                              manager=manager,
                                              container=help_screen)
    
    button_width, button_height = 100, 50
    back_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(20, 20, button_width, button_height),
                                               text='< Back',
                                               manager=manager,
                                               object_id='#back_button',
                                               container=help_screen)
    
    while run:
        dt = clock.tick(FPS)/1000

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                break

            if event.type == pygame_gui.UI_BUTTON_PRESSED and event.ui_element == back_button:
                run = False
                help_screen.hide()
                screen.show()
                break

            manager.process_events(event)

        WIN.fill(bg_color)
        manager.update(dt)
        manager.draw_ui(WIN)
        pygame.display.update()

# run sim
def run_sim(points, ball, back_button):
    run = True
    back = False
    vectors = []
    thicknesses = []
    
    points = list(filter(lambda sublist: len(sublist) >= 3, points))
    new_points = []
    for sublist in points:
        thicknesses.append(sublist[0][0])
        point_list = np.array(sublist[1:])
        point_list = point_list[np.append(np.any(np.diff(point_list, axis=0) != [0, 0], axis=1), True)]
        new_points.append(point_list)
        vectors.append(np.diff(point_list, axis=0))

    while run:
        dt = clock.tick(FPS)/1000

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                break

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                run = False
                break
            
            elif event.type == pygame_gui.UI_BUTTON_PRESSED and event.ui_element == back_button:
                run = False
                back = True
                break
            
            manager.process_events(event)

        # physics
        candidates = []
        center = np.array([ball.x, ball.y])
        
        if not ball.freefall:
            curve_idx, seg_idx = ball.checkpoint
            start = max(seg_idx - 20, 0)
            end = min(seg_idx + 21, len(vectors[curve_idx]))
            vecs = vectors[curve_idx][start: end]
            projs = center - new_points[curve_idx][start: end]

            scalars = np.einsum('ij, ij -> i', vecs, projs) / np.einsum('ij, ij -> i', vecs, vecs)
            projections = projs - vecs * scalars[:, np.newaxis]
            distances = np.linalg.norm(projections, axis=1)
            on_line = np.sum(projs * vecs, axis=1) / np.sum(vecs * vecs, axis=1)
            mask = (distances <= ball.radius + thicknesses[curve_idx]/2) & (on_line >= 0) & (on_line <= 1)
            indices = np.where(mask)[0]

            if len(indices):
                print('local search successful!')
                diffs = np.abs(indices + start - seg_idx)
                ball.checkpoint = np.array([curve_idx, start + np.where(diffs == np.min(diffs))[0][0]])
                ball.tangent_vector = vectors[ball.checkpoint[0]][ball.checkpoint[1]]
                projection = projections[ball.checkpoint[1] - start]

                if projection[1] >= 0:
                    ball.on_top = True
                else:
                    ball.on_top = False

                if distances[ball.checkpoint[1] - start] < ball.radius + thicknesses[curve_idx]/2:
                    tangent_point = center - projection
                    ball.x, ball.y = tangent_point + projection * (ball.radius + thicknesses[curve_idx]) / np.linalg.norm(projection)
            else:
                ball.freefall = True

        if ball.freefall:
            proj_vectors = [center - sublist[:-1] for sublist in new_points]
            for i in range(len(vectors)):
                scalars = np.einsum('ij, ij -> i', vectors[i], proj_vectors[i]) / np.einsum('ij, ij -> i', vectors[i], vectors[i])
                projections = proj_vectors[i] - vectors[i] * scalars[:, np.newaxis]
                distances = np.linalg.norm(projections, axis=1)
                on_line = np.sum(proj_vectors[i] * vectors[i], axis=1) / np.sum(vectors[i] * vectors[i], axis=1)
                mask = (distances <= ball.radius + thicknesses[i]/2) & (on_line >= 0) & (on_line <= 1)

                indices = np.where(mask)[0]
                if len(indices):
                    ball.freefall = False
                    ball.checkpoint = np.array([i, indices[0]])
                    ball.tangent_vector = vectors[ball.checkpoint[0]][ball.checkpoint[1]]
                    projection = projections[ball.checkpoint[1]]

                    if projection[1] >= 0:
                        ball.on_top = True
                    else:
                        ball.on_top = False
                    
                    if distances[ball.checkpoint[1]] < ball.radius + thicknesses[i]/2:
                        tangent_point = center - projection
                        ball.x, ball.y = tangent_point + projection * (ball.radius + thicknesses[i]) / np.linalg.norm(projection)

                    break

        ball.move(dt)

        if ball.x + ball.radius <= 0 or ball.x - ball.radius >= WIDTH or ball.y + ball.radius <= 0 or ball.y - ball.radius >= HEIGHT:
            run = False
            ball.x = 0
            ball.y = 0
            ball.xvel = 0
            ball.yvel = 0

        # draw
        WIN.fill(bg_color)
        for sublist in points:
            pygame.draw.lines(WIN, sublist[0][1], False, sublist[1:], sublist[0][0])
            
        pygame.draw.circle(WIN, ball.color, (ball.x, ball.y), ball.radius)

        manager.update(dt)
        manager.draw_ui(WIN)
        pygame.display.update()
    
    return ball, back

# edit sim
def edit():
    run = True
    points = []
    ball = Ball(20, (0, 0), (0, 0), pygame.Color(255, 255, 255))

    # UI elements
    coords = pygame_gui.elements.UITextBox(relative_rect=pygame.Rect((20, 730), (75, 60)), object_id='#coords',
                                         manager=manager, html_text=f'X: {pygame.mouse.get_pos()[0]}<br>Y: {pygame.mouse.get_pos()[1]}')
    edit_screen = pygame_gui.elements.UIPanel(relative_rect=pygame.Rect(WIDTH - 200, 0, 200, HEIGHT), manager=manager, object_id='#edit_screen')
    back_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(20, 20, 100, 50),
                                               text='< Back', manager=manager,
                                               object_id='#back_button')
    hide_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(20, 80, 125, 50),
                                               text='Hide Toolbar', manager=manager,
                                               object_id="#hide_button")
    
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect(20, 20, 100, 30), text='Tool:',
                                                  manager=manager, container=edit_screen)
    tool_select = pygame_gui.elements.UIDropDownMenu(relative_rect=pygame.Rect(20, 50, 150, 50),
                                                          options_list=['Pencil', 'Eraser'], starting_option='Pencil',
                                                          manager=manager, container=edit_screen)
    
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect(20, 120, 200, 30), text='Tool Size (px):',
                                manager=manager, container=edit_screen)
    size_select = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect(20, 150, 100, 40), initial_text='3',
                                                      manager=manager, container=edit_screen)
    size_select.set_allowed_characters('numbers')
    
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect(20, 210, 150, 30), text='Color:',
                                manager=manager, container=edit_screen)
    color_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(100, 210, 30, 30), text='',
                                 manager=manager, container=edit_screen, object_id='#color_button')
    color_picker_open = False
    selected_color = pygame.Color(255, 255, 255)
    color_select = None

    pygame_gui.elements.UILabel(relative_rect=pygame.Rect(20, 260, 200, 30), text='Ball Radius (px):',
                                manager=manager, container=edit_screen)
    radius_select = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect(20, 290, 100, 40), initial_text='20',
                                                        manager=manager, container=edit_screen)
    radius_select.set_allowed_characters(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.'])
    
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect(20, 350, 200, 30), text='Ball Color:',
                                manager=manager, container=edit_screen)
    ball_color_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(140, 350, 30, 30), text='',
                                                     manager=manager, container=edit_screen, object_id='#ball_color_button')
    ball_color = pygame.Color(255, 255, 255)
    ball_color_select = None

    pygame_gui.elements.UILabel(relative_rect=pygame.Rect(20, 400, 200, 30), text='Initial Position:',
                                manager=manager, container=edit_screen)
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect(20, 430, 30, 30), text='X:',
                                manager=manager, container=edit_screen, object_id='#caption')
    x_pos = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect(45, 430, 50, 30), initial_text='0',
                                                manager=manager, container=edit_screen, object_id='#caption')
    x_pos.set_allowed_characters(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.'])
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect(110, 430, 30, 30), text='Y:',
                                manager=manager, container=edit_screen, object_id='#caption')
    y_pos = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect(135, 430, 50, 30), initial_text='0',
                                                manager=manager, container=edit_screen, object_id='#caption')
    y_pos.set_allowed_characters(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.'])
    
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect(20, 480, 200, 30), text='Velocity (px/s):',
                                manager=manager, container=edit_screen)
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect(20, 510, 30, 30), text='X:',
                                manager=manager, container=edit_screen, object_id='#caption')
    x_vel = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect(45, 510, 50, 30), initial_text='0',
                                                manager=manager, container=edit_screen, object_id='#caption')
    x_vel.set_allowed_characters(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '.'])
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect(110, 510, 30, 30), text='Y:',
                                manager=manager, container=edit_screen, object_id='#caption')
    y_vel = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect(135, 510, 50, 30), initial_text='0',
                                                manager=manager, container=edit_screen, object_id='#caption')
    y_vel.set_allowed_characters(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '.'])

    pygame_gui.elements.UILabel(relative_rect=pygame.Rect(20, 560, 200, 30), text='Constants:',
                                manager=manager, container=edit_screen)
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect(20, 590, 75, 30), text='Friction:',
                                manager=manager, container=edit_screen, object_id='#caption')
    friction_select = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect(85, 590, 50, 30), initial_text='0',
                                                          manager=manager, container=edit_screen, object_id='#caption')
    friction_select.set_allowed_characters(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.'])
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect(140, 590, 60, 30), text='× Fₙ',
                                manager=manager, container=edit_screen, object_id='#symbol')
    
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect(20, 625, 75, 30), text='Gravity:',
                                manager=manager, container=edit_screen, object_id='#caption')
    gravity_select = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect(85, 625, 65, 30), initial_text='1000',
                                                         manager=manager, container=edit_screen, object_id='#caption')
    gravity_select.set_allowed_characters(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.'])
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect(155, 625, 60, 30), text='px/s²',
                                manager=manager, container=edit_screen, object_id='#symbol')

    pygame_gui.elements.UILabel(relative_rect=pygame.Rect(20, 660, 100, 30), text='Air density:',
                                manager=manager, container=edit_screen, object_id='#caption')
    density_select = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect(110, 660, 40, 30), initial_text='0',
                                                         manager=manager, container=edit_screen, object_id='#caption')
    density_select.set_allowed_characters(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.'])
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect(152, 660, 60, 30), text='kg/px³',
                                manager=manager, container=edit_screen, object_id='#symbol')

    validated = True
    
    while run:
        dt = clock.tick(FPS)/1000

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
            
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN and not(color_picker_open):
                edit_screen.hide()
                hide_button.hide()
                coords.hide()

                ball, back = run_sim(points, ball, back_button)
                if back:
                    back_button.hide()
                    menu()
                else:
                    x_pos.set_text(f'{round(ball.x, 1)}')
                    y_pos.set_text(f'{round(ball.y, 1)}')
                    x_vel.set_text(f'{round(ball.xvel, 1)}')
                    y_vel.set_text(f'{round(ball.yvel, 1)}')
                    edit_screen.show()
                    hide_button.show()
                    coords.show()
            
            if event.type == pygame.MOUSEMOTION:
                coords.set_text(f'X: {pygame.mouse.get_pos()[0]}<br>Y: {pygame.mouse.get_pos()[1]}')

            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == back_button and not(color_picker_open):
                    edit_screen.hide()
                    back_button.hide()
                    hide_button.hide()
                    coords.hide()
                    menu()

                elif event.ui_element == hide_button:
                    if edit_screen.visible:
                        edit_screen.hide()
                        hide_button.set_text('Show Toolbar')
                    else:
                        edit_screen.show()
                        hide_button.set_text('Hide Toolbar')
                
                elif event.ui_element == color_button and not(color_picker_open):
                    color_picker_open = True
                    color_select = pygame_gui.windows.UIColourPickerDialog(pygame.Rect(200, 200, 500, 500), manager=manager, window_title='Color Picker',
                                                                           initial_colour=selected_color, object_id='#color_picker')
                elif event.ui_element == ball_color_button and not(color_picker_open):
                    color_picker_open = True
                    ball_color_select = pygame_gui.windows.UIColourPickerDialog(pygame.Rect(200, 200, 500, 500), manager=manager, window_title='Ball Color Picker',
                                                                                initial_colour=ball_color, object_id='#color_picker')
                    
            if event.type == pygame_gui.UI_COLOUR_PICKER_COLOUR_PICKED:
                if event.ui_element == color_select:
                    color_picker_open = False
                    selected_color = event.colour
                    hex_color = "#{:02x}{:02x}{:02x}".format(selected_color.r, selected_color.g, selected_color.b)
                    hover_color = hex_color
                    if selected_color.r >= 20 and selected_color.g >= 20 and selected_color.b >= 20:
                        hover_color = "#{:02x}{:02x}{:02x}".format(selected_color.r - 20, selected_color.g - 20, selected_color.b - 20)

                    with open('theme.json') as theme:
                        theme_data = json.load(theme)
                        theme_data['#color_button']['colors']['normal_bg'] = hex_color
                        theme_data['#color_button']['colors']['hovered_bg'] = hover_color
                    
                    with open('theme.json', 'w') as theme:
                        json.dump(theme_data, theme, indent=2)

                elif event.ui_element == ball_color_select:
                    color_picker_open = False
                    ball_color = event.colour
                    ball.color = ball_color
                    hex_color = "#{:02x}{:02x}{:02x}".format(ball_color.r, ball_color.g, ball_color.b)
                    hover_color = hex_color
                    if ball_color.r >= 20 and ball_color.g >= 20 and ball_color.b >= 20:
                        hover_color = "#{:02x}{:02x}{:02x}".format(ball_color.r - 20, ball_color.g - 20, ball_color.b - 20)

                    with open('theme.json') as theme:
                        theme_data = json.load(theme)
                        theme_data['#ball_color_button']['colors']['normal_bg'] = hex_color
                        theme_data['#ball_color_button']['colors']['hovered_bg'] = hover_color
                    
                    with open('theme.json', 'w') as theme:
                        json.dump(theme_data, theme, indent=2)


            elif event.type == pygame_gui.UI_WINDOW_CLOSE and event.ui_object_id == '#color_picker':
                color_picker_open = False

            elif event.type == pygame_gui.UI_TEXT_ENTRY_CHANGED:
                validated = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if not validated:
                    validated = True
                    
                    # tool size
                    if size_select.get_text() == '':
                        size_select.set_text('3')
                
                    size = int(size_select.get_text())
                    if size < 1 or size > 10:
                        size = 3
                        size_select.set_text('3')
                
                    # ball radius
                    if radius_select.get_text() == '' or list(radius_select.get_text()).count('.') > 1:
                        radius_select.set_text('20')
                    
                    radius = float(radius_select.get_text())
                    if radius < 10 or radius > 75:
                        radius = 20
                        radius_select.set_text('20')
                    ball.radius = radius

                    # initial position
                    if x_pos.get_text() == '' or list(x_pos.get_text()).count('.') > 1:
                        x_pos.set_text('0')
                    
                    X = float(x_pos.get_text())
                    if X < 0 or X > WIDTH:
                        X = 0
                        x_pos.set_text('0')
                    ball.x = X

                    if y_pos.get_text() == '' or list(y_pos.get_text()).count('.') > 1:
                        y_pos.set_text('0')

                    Y = float(y_pos.get_text())
                    if Y < 0 or Y > HEIGHT:
                        Y = 0
                        y_pos.set_text('0')
                    ball.y = Y

                    # initial velocity
                    if x_vel.get_text() == '' or list(x_vel.get_text()).count('.') > 1 or list(x_vel.get_text()).count('-') > 1:
                        x_vel.set_text('0')
                    
                    XVEL = float(x_vel.get_text())
                    if XVEL < -500 or XVEL > 500:
                        XVEL = 0
                        x_vel.set_text('0')
                    ball.xvel = XVEL

                    if y_vel.get_text() == '' or list(y_vel.get_text()).count('.') > 1 or list(y_vel.get_text()).count('-') > 1:
                        y_vel.set_text('0')

                    YVEL = float(y_vel.get_text())
                    if YVEL < -500 or YVEL > 500:
                        YVEL = 0
                        y_vel.set_text('0')
                    ball.yvel = YVEL

                    # friction
                    if friction_select.get_text() == '' or list(friction_select.get_text()).count('.') > 1:
                        friction_select.set_text('0')
                    
                    global friction
                    friction = float(friction_select.get_text())

                    # gravity
                    if gravity_select.get_text() == '' or list(gravity_select.get_text()).count('.') > 1:
                        gravity_select.set_text('1000')
                    
                    GRAV = float(gravity_select.get_text())
                    if GRAV > 1500:
                        GRAV = 1000
                        gravity_select.set_text('1000')
                    global g
                    g = GRAV

                    # air density
                    if density_select.get_text() == '' or list(density_select.get_text()).count('.') > 1:
                        density_select.set_text('0')
                    
                    global rho
                    rho = float(density_select.get_text())

                if tool_select.selected_option[0] == 'Pencil':
                    points.append([[int(size_select.get_text()), pygame.Color(selected_color.r, selected_color.g, selected_color.b)]])

            manager.process_events(event)

        # draw
        WIN.blit(GRID)
        pygame.draw.circle(WIN, ball.color, (ball.x, ball.y), ball.radius)

        if ((pygame.mouse.get_pressed()[0] and edit_screen.visible == False) or (pygame.mouse.get_pressed()[0] and pygame.mouse.get_pos()[0] < 800)) and not(color_picker_open):
            if tool_select.selected_option[0] == 'Pencil':
                points[-1].append(pygame.mouse.get_pos())

            elif tool_select.selected_option[0] == 'Eraser':
                new_points = []
                for sublist in points:
                    if len(sublist) <= 1:
                        continue

                    splice_points = []
                    radius = int(size_select.get_text()) * 5
                    point_list = np.array(sublist[1:])
                    is_in_range = np.linalg.norm(point_list - pygame.mouse.get_pos(), axis=1) <= radius
                    diffs = np.diff(is_in_range.astype(int))

                    if not(is_in_range[0]):
                        splice_points.append([0])

                    for i in range(len(diffs)):
                        if diffs[i] == 1:
                            splice_points[-1].append(i+1)
                        elif diffs[i] == -1:
                            splice_points.append([i+1])
                    
                    if not(is_in_range[-1]):
                        splice_points[-1].append(len(point_list))
                    
                    for splice in splice_points:
                        new_sublist = list(point_list[splice[0]:splice[1]])
                        new_sublist.insert(0, sublist[0])
                        new_points.append(new_sublist)

                points = new_points
  
        for sublist in points:
            if len(sublist) >= 3:
                pygame.draw.lines(WIN, sublist[0][1], False, sublist[1:], sublist[0][0])

        manager.update(dt)
        manager.draw_ui(WIN)
        pygame.display.update()
    
    pygame.quit()

# menu screen
def menu():
    run = True
    
    menu_screen = pygame_gui.elements.UIPanel(relative_rect=pygame.Rect(0, 0, WIDTH, HEIGHT), manager=manager)

    # buttons
    button_width, button_height = 200, 75
    start_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((WIDTH - button_width)/2, 300, button_width, button_height),
                                                text='Start', manager=manager, container=menu_screen)
    help_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((WIDTH - button_width)/2, 400, button_width, button_height),
                                               text='Help', manager=manager, container=menu_screen)

    while run:
        dt = clock.tick(FPS)/1000

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                menu_screen.hide()

                if event.ui_element == help_button:
                    help(menu_screen)

                if event.ui_element == start_button:
                    edit()
        
            manager.process_events(event)
        
        # drawing
        WIN.fill(bg_color)
        title = TITLE_FONT.render('Physics Sandbox', 1, 'white')
        WIN.blit(title, ((WIDTH - title.get_width())/2, 50))

        manager.update(dt)
        manager.draw_ui(WIN)
        pygame.display.update()

    pygame.quit()
    
menu()