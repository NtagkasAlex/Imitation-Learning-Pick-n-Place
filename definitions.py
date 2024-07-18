from PandaEnv import MyCustomPandaEnv  # Replace with the actual import

def panda_env():
    return MyCustomPandaEnv(
        render_mode="rgb_array",
        renderer="OpenGL",
        render_width=200,
        render_height=200,
        render_target_position=[0.2, 0., 0.1],
        render_distance=0.1,
        render_yaw=90,
        render_pitch=-20,
        render_roll=0,
        reward_type="sparse",
        control_type="ee"
    )