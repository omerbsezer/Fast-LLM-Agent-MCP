import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# --- Constants ---
# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Space Shuttle Game"
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Player constants
PLAYER_WIDTH = 50
PLAYER_HEIGHT = 50
PLAYER_SPEED = 7
PLAYER_INITIAL_BOTTOM_OFFSET = 10
PLAYER_TOP_BOUND = SCREEN_HEIGHT // 2  # Limit vertical movement to top half of the screen

# Meteor constants
METEOR_WIDTH = 30
METEOR_HEIGHT = 30
METEOR_MIN_SPEED = 3
METEOR_MAX_SPEED = 7
METEOR_SPAWN_OFFSET_Y_MIN = -100
METEOR_SPAWN_OFFSET_Y_MAX = -40
METEOR_SPAWN_INTERVAL_MS = 1000  # Spawn a meteor every 1 second

# Bullet constants
BULLET_WIDTH = 5
BULLET_HEIGHT = 15
BULLET_SPEED = -10  # Moves upwards

# Game constants
SCORE_INCREMENT = 10
FONT_SIZE = 36
GAME_OVER_MESSAGE = "GAME OVER! Press Q to Quit"

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(SCREEN_TITLE)

# Player (Shuttle) Class
class Player(pygame.sprite.Sprite):
    """Represents the player's shuttle, controlled by arrow keys."""
    def __init__(self):
        super().__init__()
        # Visuals: Currently uses a colored square. For a more complete game, consider loading an actual image:
        # self.image = pygame.image.load("assets/shuttle.png").convert_alpha()
        self.image = pygame.Surface([PLAYER_WIDTH, PLAYER_HEIGHT])
        self.image.fill(BLUE)
        self.rect = self.image.get_rect()
        self.rect.centerx = SCREEN_WIDTH // 2
        self.rect.bottom = SCREEN_HEIGHT - PLAYER_INITIAL_BOTTOM_OFFSET
        self.speed = PLAYER_SPEED

    def update(self):
        """Updates the player's position based on key presses and keeps it within defined screen bounds."""
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.rect.x -= self.speed
        if keys[pygame.K_RIGHT]:
            self.rect.x += self.speed
        if keys[pygame.K_UP]:
            self.rect.y -= self.speed
        if keys[pygame.K_DOWN]:
            self.rect.y += self.speed

        # Keep player within screen bounds
        self.rect.left = max(0, self.rect.left)
        self.rect.right = min(SCREEN_WIDTH, self.rect.right)
        self.rect.top = max(PLAYER_TOP_BOUND, self.rect.top)
        self.rect.bottom = min(SCREEN_HEIGHT, self.rect.bottom)

    def shoot(self):
        """Creates and adds a bullet to the appropriate sprite groups."""
        bullet = Bullet(self.rect.centerx, self.rect.top)
        all_sprites.add(bullet)
        bullets.add(bullet)

# Meteor Class
class Meteor(pygame.sprite.Sprite):
    """Represents an incoming meteor falling from the top of the screen."""
    def __init__(self):
        super().__init__()
        # Visuals: Currently uses a colored square. For a more complete game, consider loading an actual image:
        # self.image = pygame.image.load("assets/meteor.png").convert_alpha()
        self.image = pygame.Surface([METEOR_WIDTH, METEOR_HEIGHT])
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.rect.x = random.randrange(0, SCREEN_WIDTH - self.rect.width)
        self.rect.y = random.randrange(METEOR_SPAWN_OFFSET_Y_MIN, METEOR_SPAWN_OFFSET_Y_MAX)
        self.speed_y = random.randrange(METEOR_MIN_SPEED, METEOR_MAX_SPEED)

    def update(self):
        """Moves the meteor downwards and removes it if it goes off-screen."""
        self.rect.y += self.speed_y
        if self.rect.top > SCREEN_HEIGHT:
            self.kill()

# Bullet Class
class Bullet(pygame.sprite.Sprite):
    """Represents a bullet fired by the player."""
    def __init__(self, x, y):
        super().__init__()
        # Visuals: Currently uses a colored rectangle. For a more complete game, consider loading an actual image:
        # self.image = pygame.image.load("assets/bullet.png").convert_alpha()
        self.image = pygame.Surface([BULLET_WIDTH, BULLET_HEIGHT])
        self.image.fill(YELLOW)
        self.rect = self.image.get_rect()
        self.rect.centerx = x
        self.rect.bottom = y
        self.speed_y = BULLET_SPEED

    def update(self):
        """Moves the bullet upwards and removes it if it goes off-screen."""
        self.rect.y += self.speed_y
        if self.rect.bottom < 0:
            self.kill()

# Game Variables
score = 0
font = pygame.font.Font(None, FONT_SIZE)

# Sprite Groups
all_sprites = pygame.sprite.Group()
meteors = pygame.sprite.Group()
bullets = pygame.sprite.Group()

# Create player
player = Player()
all_sprites.add(player)

# Game Loop
running = True
clock = pygame.time.Clock()

# Meteor spawning timer
meteor_spawn_event = pygame.USEREVENT + 1
pygame.time.set_timer(meteor_spawn_event, METEOR_SPAWN_INTERVAL_MS)

game_over = False

while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and not game_over:
                player.shoot()
            # Handle 'Q' key press for quitting when game is over
            elif event.key == pygame.K_q and game_over:
                running = False
        if event.type == meteor_spawn_event and not game_over:
            new_meteor = Meteor()
            all_sprites.add(new_meteor)
            meteors.add(new_meteor)

    if not game_over:
        # Update game state
        all_sprites.update()

        # Check for collisions: Bullet vs. Meteor
        hits = pygame.sprite.groupcollide(bullets, meteors, True, True)
        for hit in hits:
            score += SCORE_INCREMENT

        # Check for collisions: Player vs. Meteor
        player_hits = pygame.sprite.spritecollide(player, meteors, False)
        if player_hits:
            game_over = True

    # Drawing
    screen.fill(BLACK)  # Background
    all_sprites.draw(screen)

    # Display Score
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

    # Game Over screen
    if game_over:
        game_over_text = font.render(GAME_OVER_MESSAGE, True, WHITE)
        text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        screen.blit(game_over_text, text_rect)
        # The 'Q' key for quitting is now handled in the main event loop,
        # so no separate event loop is needed here.

    pygame.display.flip()  # Update the display

    # Cap the frame rate
    clock.tick(FPS)

pygame.quit()
sys.exit()