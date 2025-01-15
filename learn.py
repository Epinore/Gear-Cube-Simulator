
import pygame
import OpenGL.GL as GL
import OpenGL.GLU as GLU
from PIL import Image
import os
import math

# --------------------- Cube and Rubik Classes ---------------------

class Cube(object):
    # Define the edges of the cube
    edges = (
        (0, 1), (0, 3), (0, 4), (2, 1), (2, 3), (2, 7),
        (6, 3), (6, 4), (6, 7), (5, 1), (5, 4), (5, 7)
    )

    # Define the polygons (faces) of the cube
    polygons = (
        (0, 1, 2, 3),  # Back
        (3, 2, 7, 6),  # Left
        (6, 7, 5, 4),  # Front
        (4, 5, 1, 0),  # Right
        (1, 5, 7, 2),  # Top
        (4, 0, 3, 6)   # Bottom
    )

    # Define the vertices of the cube
    vertices = (
        (1, -1, -1),  # 0
        (1, 1, -1),   # 1
        (-1, 1, -1),  # 2
        (-1, -1, -1), # 3
        (1, -1, 1),   # 4
        (1, 1, 1),    # 5
        (-1, -1, 1),  # 6
        (-1, 1, 1)    # 7
    )

    def __init__(self, ident: tuple, n: int, scale: int, face_textures: list) -> None:
        """
        Initialize the Cube object with per-face textures.
        :param ident: Position tuple (x, y, z)
        :param n: Rubik's cube order (e.g., 3)
        :param scale: Scale factor
        :param face_textures: List of 6 texture image paths or None for each face
        """
        self.n = n
        self.scale = scale
        self.ident = ident  # Original position of the cube
        self.current = list(ident)  # Current position of the cube
        self.rot = [[1 if i == j else 0 for i in range(3)] for j in range(3)]
        # Initialize the rotation matrix

        # Initialize texture rotation angles for each face based on position
        self.texture_rotations = self.set_texture_rotations()

        # Load textures for each face
        self.texture_ids = [None] * 6  # One texture ID per face
        for i in range(6):
            if face_textures[i]:
                self.texture_ids[i] = self.load_texture(os.path.join("image", face_textures[i]))
            else:
                self.texture_ids[i] = None  # No texture for this face

        # Initialize self-rotation attributes
        self.self_rotation = None

    def set_texture_rotations(self):
        """
        Set the texture rotation angles for each face based on the cube's position and type.
        Returns a list of six integers corresponding to the back, left, front, right, top, and bottom faces.
        """
        x, y, z = self.current
        rotations = [0, 0, 0, 0, 0, 0]  # Back, Left, Front, Right, Top, Bottom

        # Determine cube type
        count = sum(coord in [0, 2] for coord in self.current)
        is_corner = count == 3
        is_edge = count == 2
        is_center = count == 1

        if is_corner:
            # Set rotations for corner cubes
            if z == 2:
                if x == 0 and y == 2:  # Front-Top-Left
                    rotations = [0, 3, 2, 0, 3, 0]
                elif x == 2 and y == 2:  # Front-Top-Right
                    rotations = [0, 0, 3, 2, 2, 0]
                elif x == 0 and y == 0:  # Front-Bottom-Left
                    rotations = [0, 0, 1, 0, 0, 0]
                elif x == 2 and y == 0:  # Front-Bottom-Right
                    rotations = [0, 0, 0, 1, 0, 1]

            elif z == 0:
                if x == 0 and y == 2:  # Back-Top-Left
                    rotations = [3, 2, 0, 0, 0, 0]
                elif x == 2 and y == 2:  # Back-Top-Right
                    rotations = [2, 0, 0, 3, 1, 0]
                elif x == 0 and y == 0:  # Back-Bottom-Left
                    rotations = [0, 1, 0, 0, 0, 3]
                elif x == 2 and y == 0:  # Back-Bottom-Right
                    rotations = [1, 0, 0, 0, 0, 2]

        elif is_edge:
            # Set rotations for edge cubes
            if (x, y, z) == (1, 2, 2):  # Front-Top
                rotations = [0, 0, 2, 0, 2, 0]
            elif (x, y, z) == (2, 1, 2):  # Front-Right
                rotations = [0, 0, 3, 1, 0, 0]
            elif (x, y, z) == (1, 0, 2):  # Front-Bottom
                rotations = [0, 0, 0, 0, 0, 0]
            elif (x, y, z) == (0, 1, 2):  # Front-Left
                rotations = [0, 3, 1, 0, 0, 0]
            elif (x, y, z) == (1, 2, 0):  # Back-Top
                rotations = [2, 0, 0, 0, 0, 0]
            elif (x, y, z) == (2, 1, 0):  # Back-Right
                rotations = [1, 0, 0, 3, 0, 0]
            elif (x, y, z) == (1, 0, 0):  # Back-Bottom
                rotations = [0, 0, 0, 0, 0, 2]
            elif (x, y, z) == (0, 1, 0):  # Back-Left
                rotations = [3, 1, 0, 0, 0, 0]
            elif (x, y, z) == (2, 2, 1):  # Top-Right
                rotations = [0, 0, 0, 2, 1, 0]
            elif (x, y, z) == (0, 2, 1):  # Top-Left
                rotations = [0, 2, 0, 0, 3, 0]
            elif (x, y, z) == (2, 0, 1):  # Bottom-Right
                rotations = [0, 0, 0, 0, 0, 1]
            elif (x, y, z) == (0, 0, 1):  # Bottom-Left
                rotations = [0, 0, 0, 0, 0, 3]

        # For center cubes, rotations remain unchanged (all faces not rotated)

        return rotations

    def is_affected(self, axis: int, slc: int):
        return self.current[axis] == slc

    def update(self, axis: int, slc: int, dr: int, degrees: int):
        """
        Update the rotation matrix and the cube's position.
        :param axis: Rotation axis (0: x, 1: y, 2: z)
        :param slc: Slice index
        :param dr: Direction of rotation (1 or -1)
        :param degrees: Degrees to rotate
        """
        if not self.is_affected(axis, slc):
            return  # Do not execute if not in the rotating layer

        i = (axis + 1) % 3
        j = (axis + 2) % 3

        # Calculate the number of 90-degree rotations
        num_rotations = degrees // 90

        for _ in range(num_rotations):
            # Update the rotation matrix, perform 90-degree rotation
            for k in range(3):
                self.rot[k][i], self.rot[k][j] = -self.rot[k][j] * dr, self.rot[k][i] * dr

            # Update the small cube's position coordinates
            self.current[i], self.current[j] = (
                self.current[j] if dr < 0 else self.n - 1 - self.current[j],
                self.current[i] if dr > 0 else self.n - 1 - self.current[i]
            )

    def transform_matrix(self):
        s_a = [[s * self.scale for s in a] for a in self.rot]
        s_t = [(p - (self.n - 1) / 2) * 2 * self.scale for p in self.current]
        return [
            *s_a[0], 0,
            *s_a[1], 0,
            *s_a[2], 0,
            *s_t, 1
        ]

    def load_texture(self, image_path):
        try:
            # Load image and flip vertically
            img = Image.open(image_path).convert("RGBA")  # Use RGBA to support transparency
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img_data = img.tobytes("raw", "RGBA", 0, -1)  # Use RGBA format
            width, height = img.size

            texture_id = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height,
                0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_data
            )
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)  # Generate mipmaps (optional)
            return texture_id

        except Exception as e:
            print(f"Failed to load texture {image_path}: {e}")
            return None

    def draw(self, polygons, active_rotations):
        GL.glPushMatrix()
        # Apply all active rotations that affect this cube
        for rotation in active_rotations:
            axis = rotation['axis']
            slc = rotation['slc']
            dr = rotation['dr']
            current_angle = rotation['current_angle']

            if self.is_affected(axis, slc):
                # Set the rotation axis vector
                rotation_axis = [0, 0, 0]
                rotation_axis[axis] = 1  # Set the rotation axis (x, y, or z)
                GL.glRotatef(current_angle, *rotation_axis)

        # Apply self-rotation if any
        if self.self_rotation is not None:
            axis_point_scaled = [(coord - (self.n - 1) / 2) * 2 * self.scale for coord in self.self_rotation['axis_point']]
            GL.glTranslatef(*[-coord for coord in axis_point_scaled])
            GL.glRotatef(self.self_rotation['current_angle'], *self.self_rotation['axis'])
            GL.glTranslatef(*axis_point_scaled)

        GL.glMultMatrixf(self.transform_matrix())

        # Draw each face with its corresponding texture
        for i, face in enumerate(polygons):
            if self.texture_ids[i]:
                # Enable texture
                GL.glEnable(GL.GL_TEXTURE_2D)
                GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_ids[i])
                GL.glColor3f(1, 1, 1)  # Set color to white (normalized)
            else:
                # If texture not loaded, use color fill
                GL.glDisable(GL.GL_TEXTURE_2D)
                GL.glColor3fv((1, 1, 1))  # Set to white

            # Draw cube surfaces and apply texture
            GL.glBegin(GL.GL_QUADS)
            if self.texture_ids[i]:
                # Get texture coordinates based on each face's rotation
                rotation = self.texture_rotations[i]
                tex_coords = self.get_face_tex_coords(rotation)
                for j, vertex in enumerate(face):
                    GL.glTexCoord2fv(tex_coords[j])
                    GL.glVertex3fv(Cube.vertices[vertex])
            else:
                # If not using texture, use default color
                for vertex in face:
                    GL.glVertex3fv(Cube.vertices[vertex])
            GL.glEnd()

        # Draw cube edges
        GL.glEnable(GL.GL_LINE_SMOOTH)
        GL.glLineWidth(3)
        GL.glColor3fv((1, 1, 1))  # Set edge color to white (normalized)
        GL.glBegin(GL.GL_LINES)
        for edge in Cube.edges:
            for vertex in edge:
                GL.glVertex3fv(Cube.vertices[vertex])
        GL.glEnd()
        GL.glDisable(GL.GL_LINE_SMOOTH)

        GL.glPopMatrix()

    def get_face_tex_coords(self, rotation: int):
        """
        Get texture coordinates for each face based on rotation angle.
        :param rotation: Multiple of 90 degrees (0, 1, 2, 3)
        """
        # Base texture coordinates
        tex_coords = [
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)
        ]

        if rotation == 0:
            return tex_coords
        elif rotation == 1:
            return [tex_coords[3], tex_coords[0], tex_coords[1], tex_coords[2]]
        elif rotation == 2:
            return [tex_coords[2], tex_coords[3], tex_coords[0], tex_coords[1]]
        elif rotation == 3:
            return [tex_coords[1], tex_coords[2], tex_coords[3], tex_coords[0]]
        else:
            return tex_coords

    def set_face_rotation(self, face_index: int, rotation: int):
        """
        Set the texture rotation angle for a specific face.
        :param face_index: Index of the face (0-5)
        :param rotation: Multiple of 90 degrees (0, 1, 2, 3)
        """
        if 0 <= face_index < 6 and 0 <= rotation < 4:
            self.texture_rotations[face_index] = rotation
        else:
            print("Invalid face index or rotation value.")


class Rubik:
    def __init__(self, scale):
        """
        Initialize the Rubik's Cube.
        :param scale: Scale factor
        """
        self.n = 3
        cr = range(self.n)
        self.scale = scale
        self.cubes = self.init_cube(cr)
        self.cubes_dict = {(cube.current[0], cube.current[1], cube.current[2]): cube for cube in self.cubes}
        self.cubes_ident_dict = {cube.ident: cube for cube in self.cubes}

    def init_cube(self, cr):
        """
        Initialize all small cubes in the Rubik's Cube with per-face textures.
        :param cr: Coordinate range
        :return: List of Cube objects
        """
        cubes = []
        # Mapping from face index to texture number
        # 0: Back (M4, S4, T4)
        # 1: Left (M6, S6, T6)
        # 2: Front (M2, S2, T2)
        # 3: Right (M3, S3, T3)
        # 4: Top (M1, S1, T1)
        # 5: Bottom (M5, S5, T5)
        face_n_map = {0:4, 1:6, 2:2, 3:3, 4:1, 5:5}

        for z in cr:
            for y in cr:
                for x in cr:
                    # Exclude the overall center cube
                    if (x, y, z) == (1, 1, 1):
                        continue  # Do not add the overall center cube

                    # Determine the type of the current cube
                    count = sum(coord in [0, 2] for coord in (x, y, z))
                    if count == 3:
                        cube_type = 'corner'
                    elif count == 2:
                        cube_type = 'edge'
                    elif count == 1:
                        cube_type = 'center'
                    else:
                        continue  # Internal cube, already excluded

                    face_textures = [None] * 6  # Initialize textures for all 6 faces

                    for face_index in range(6):
                        # Determine if the cube is on the current face
                        if face_index == 0 and z == 0:  # Back face
                            n = face_n_map[face_index]
                            if cube_type == 'corner':
                                texture_name = f'T{n}.png'
                            elif cube_type == 'edge':
                                texture_name = f'S{n}.png'
                            elif cube_type == 'center':
                                texture_name = f'M{n}.png'
                            face_textures[face_index] = texture_name
                        elif face_index == 1 and x == 0:  # Left face
                            n = face_n_map[face_index]
                            if cube_type == 'corner':
                                texture_name = f'T{n}.png'
                            elif cube_type == 'edge':
                                texture_name = f'S{n}.png'
                            elif cube_type == 'center':
                                texture_name = f'M{n}.png'
                            face_textures[face_index] = texture_name
                        elif face_index == 2 and z == 2:  # Front face
                            n = face_n_map[face_index]
                            if cube_type == 'corner':
                                texture_name = f'T{n}.png'
                            elif cube_type == 'edge':
                                texture_name = f'S{n}.png'
                            elif cube_type == 'center':
                                texture_name = f'M{n}.png'
                            face_textures[face_index] = texture_name
                        elif face_index == 3 and x == 2:  # Right face
                            n = face_n_map[face_index]
                            if cube_type == 'corner':
                                texture_name = f'T{n}.png'
                            elif cube_type == 'edge':
                                texture_name = f'S{n}.png'
                            elif cube_type == 'center':
                                texture_name = f'M{n}.png'
                            face_textures[face_index] = texture_name
                        elif face_index == 4 and y == 2:  # Top face
                            n = face_n_map[face_index]
                            if cube_type == 'corner':
                                texture_name = f'T{n}.png'
                            elif cube_type == 'edge':
                                texture_name = f'S{n}.png'
                            elif cube_type == 'center':
                                texture_name = f'M{n}.png'
                            face_textures[face_index] = texture_name
                        elif face_index == 5 and y == 0:  # Bottom face
                            n = face_n_map[face_index]
                            if cube_type == 'corner':
                                texture_name = f'T{n}.png'
                            elif cube_type == 'edge':
                                texture_name = f'S{n}.png'
                            elif cube_type == 'center':
                                texture_name = f'M{n}.png'
                            face_textures[face_index] = texture_name

                    # Create Cube object with per-face textures
                    cubie = Cube(
                        ident=(x, y, z),
                        n=3,
                        scale=self.scale,
                        face_textures=face_textures
                    )
                    cubes.append(cubie)
        return cubes

# --------------------- Main Application ---------------------

def main():
    # Initialize Rubik's Cube with a scale factor of 2
    rubik = Rubik(2)

    # Define dictionary for rotating the entire cube based on arrow key inputs
    rotate_cube = {
        pygame.K_UP: (0, -1),
        pygame.K_DOWN: (0, 1),
        pygame.K_LEFT: (1, -1),
        pygame.K_RIGHT: (1, 1)
    }

    # Define dictionary for animating and rotating individual slices based on key inputs
    # Outer slices: L, R, U, D, F, B
    # Middle slices: M, E, S
    rotate_slc_outer = {
        pygame.K_l: (0, 0, 1),   # L slice
        pygame.K_r: (0, 2, -1),  # R slice
        pygame.K_u: (1, 2, -1),  # U slice
        pygame.K_d: (1, 0, 1),   # D slice
        pygame.K_f: (2, 2, -1),  # F slice
        pygame.K_b: (2, 0, 1)    # B slice
    }

    rotate_slc_middle_mapping = {
        pygame.K_l: (0, 1, 1),   # Corresponding M slice for L
        pygame.K_r: (0, 1, -1),  # Corresponding M slice for R
        pygame.K_u: (1, 1, -1),  # Corresponding E slice for U
        pygame.K_d: (1, 1, 1),   # Corresponding E slice for D
        pygame.K_f: (2, 1, -1),  # Corresponding S slice for F
        pygame.K_b: (2, 1, 1)    # Corresponding S slice for B
    }

    # Middle slices: M, E, S can also be rotated independently by pressing their keys
    rotate_slc_middle = {
        pygame.K_m: (0, 1, -1),  # M slice
        pygame.K_e: (1, 1, 1),   # E slice
        pygame.K_s: (2, 1, -1)   # S slice
    }

    # Combine all slice dictionaries
    rotate_slc = {**rotate_slc_outer, **rotate_slc_middle}

    # Initialize active rotations list
    # Each element is a dictionary: {'axis', 'slc', 'dr', 'current_angle', 'target', 'speed'}
    active_rotations = []

    # Initialize rotation angles and cube rotation flags
    ang_x, ang_y = 0, 0
    rot_cube = [0, 0]
    running = True

    # Animation parameters
    animate_speed_primary = 4  # Degrees per frame for primary rotation (outer slices)
    animate_speed_middle = 2   # Degrees per frame for middle slice rotation

    # Initialize other variables
    zoom_factor = 1.0
    rotating = False
    rotate_start = (0, 0)

    # Helper function to normalize vectors
    def normalize_vector(v):
        length = math.sqrt(sum([coord ** 2 for coord in v]))
        return [coord / length for coord in v]

    # Main game loop
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                shift_pressed = pygame.key.get_mods() & pygame.KMOD_SHIFT
                # Check for arrow key input to rotate the entire cube
                if event.key in rotate_cube:
                    value = rotate_cube[event.key]
                    rot_cube[value[0]] = value[1]
                # Check for key input to animate and rotate individual slices
                elif event.key in rotate_slc:
                    if not active_rotations:
                        if event.key in rotate_slc_outer:
                            # Handle outer slice rotation: rotate by 180 degrees
                            axis, slc, dr = rotate_slc_outer[event.key]
                            # Add primary rotation (180 degrees)
                            active_rotations.append({
                                'axis': axis,
                                'slc': slc,
                                'dr': dr,
                                'current_angle': 0,
                                'target': 180,
                                'speed': animate_speed_primary
                            })
                            # Add corresponding middle slice rotation (90 degrees, same direction)
                            middle_axis, middle_slc, middle_dr = rotate_slc_middle_mapping[event.key]
                            active_rotations.append({
                                'axis': middle_axis,
                                'slc': middle_slc,
                                'dr': middle_dr,
                                'current_angle': 0,
                                'target': 90,
                                'speed': animate_speed_middle
                            })
                            print(f"Initiated 180-degree rotation on slice {slc} and 90-degree on corresponding middle slice")

                            # Add self-rotation for edge cubes
                            axes_pairs = []
                            if event.key == pygame.K_r or event.key == pygame.K_l:
                                # Axes for R and L moves
                                axes_pairs = [
                                    ( (1,2,2), (1,0,0) ),  # Edge pieces starting at these idents
                                    ( (1,0,2), (1,2,0) )
                                ]
                            elif event.key == pygame.K_f or event.key == pygame.K_b:
                                # For F and B moves
                                axes_pairs = [
                                    ( (0,2,1), (2,0,1) ),  # Middle-left-top to middle-right-bottom
                                    ( (0,0,1), (2,2,1) )   # Middle-left-bottom to middle-right-top
                                ]
                            elif event.key == pygame.K_u or event.key == pygame.K_d:
                                # For U and D moves
                                axes_pairs = [
                                    ( (0,1,2), (2,1,0) ),  # Front-left to back-right
                                    ( (0,1,0), (2,1,2) )   # Back-left to front-right
                                ]

                            for ident1, ident2 in axes_pairs:
                                try:
                                    cube1 = rubik.cubes_ident_dict[ident1]
                                    cube2 = rubik.cubes_ident_dict[ident2]

                                    pos1 = cube1.current
                                    pos2 = cube2.current

                                    axis_vec = [pos2[i] - pos1[i] for i in range(3)]
                                    axis_vec = normalize_vector(axis_vec)
                                    axis_point = [(pos1[i] + pos2[i]) / 2 for i in range(3)]
                                    speed = 300 / (180 / animate_speed_primary)
                                    for cube in [cube1, cube2]:
                                        cube.self_rotation = {
                                            'axis': axis_vec,
                                            'axis_point': axis_point,
                                            'current_angle': 0,
                                            'target_angle': 300,
                                            'speed': speed
                                        }
                                except KeyError:
                                    pass  # Handle cases where cubes are not found
                        elif event.key in rotate_slc_middle:
                            # Handle middle slice rotation: rotate by 180 degrees
                            axis, slc, dr = rotate_slc_middle[event.key]
                            active_rotations.append({
                                'axis': axis,
                                'slc': slc,
                                'dr': dr,
                                'current_angle': 0,
                                'target': 180,
                                'speed': animate_speed_primary
                            })
                            print(f"Initiated 180-degree rotation on middle slice {slc}")
                if event.key == pygame.K_ESCAPE:
                    running = False
            if event.type == pygame.KEYUP:
                # Reset cube rotation when arrow key is released
                if event.key in rotate_cube:
                    rot_cube = [0, 0]
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    rotating = True
                    rotate_start = pygame.mouse.get_pos()
                elif event.button == 4:  # Scroll up
                    zoom_factor = max(0.1, zoom_factor - 0.1)  # Prevent zoom from becoming too small
                elif event.button == 5:  # Scroll down
                    zoom_factor += 0.1
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    rotating = False
            if event.type == pygame.MOUSEMOTION and rotating:
                # Calculate the rotation angles based on mouse movement
                rotate_delta = pygame.mouse.get_pos()[0] - rotate_start[0], pygame.mouse.get_pos()[1] - rotate_start[1]
                ang_x += rotate_delta[1] * 0.2  # Adjust the scaling factor as needed
                ang_y += rotate_delta[0] * 0.2
                rotate_start = pygame.mouse.get_pos()

        # Update rotation angles based on arrow key input
        ang_x += rot_cube[0] * 2
        ang_y += rot_cube[1] * 2

        # Set the OpenGL modelview matrix
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glTranslatef(0, 0, -40 * zoom_factor)
        GL.glRotatef(ang_y, 0, 1, 0)
        GL.glRotatef(ang_x, 1, 0, 0)

        # Clear the screen and set background color
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        # GL.glClearColor(0, 0, 0, 1)  # Already set during initialization

        # Perform animation if active_rotations list is not empty
        if active_rotations:
            rotations_to_remove = []
            for rotation in active_rotations:
                # Increment the current angle considering rotation direction
                rotation['current_angle'] += rotation['speed'] * rotation['dr']
                if abs(rotation['current_angle']) >= rotation['target']:
                    # Clamp the angle to the target with correct sign
                    rotation['current_angle'] = rotation['target'] * (1 if rotation['dr'] > 0 else -1)
                    # Update cube slices after completing the animation
                    for cube in rubik.cubes:
                        cube.update(rotation['axis'], rotation['slc'], rotation['dr'], rotation['target'])
                    rotations_to_remove.append(rotation)
            # Remove completed rotations
            for rotation in rotations_to_remove:
                active_rotations.remove(rotation)

        # Update self-rotations for cubes
        for cube in rubik.cubes:
            if cube.self_rotation is not None:
                cube.self_rotation['current_angle'] += cube.self_rotation['speed']
                if cube.self_rotation['current_angle'] >= cube.self_rotation['target_angle']:
                    cube.self_rotation['current_angle'] = cube.self_rotation['target_angle']
                    cube.self_rotation = None

        # Draw each cube in the Rubik's Cube
        for cube in rubik.cubes:
            cube.draw(cube.polygons, active_rotations)

        # Update the display
        pygame.display.flip()
        pygame.time.wait(10)

# --------------------- Entry Point ---------------------

if __name__ == '__main__':
    # Initialize pygame and set up display
    pygame.init()
    display = (1080, 720)
    pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("Rubik's Cube Simulator")

    # Enable depth testing and set up the perspective projection matrix
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glMatrixMode(GL.GL_PROJECTION)
    GLU.gluPerspective(45, (display[0] / display[1]), 1, 100.0)

    # Set background color once (optimization: set it only once)
    GL.glClearColor(0, 0, 0, 1)  # Black background

    # Initialize variables for zoom and rotation
    zoom_factor = 1.0
    rotating = False
    rotate_start = (0, 0)

    # Start the main game loop
    main()

    # Quit pygame when the game loop exits
    pygame.quit()
