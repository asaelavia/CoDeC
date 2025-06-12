import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import threading
import queue
import pandas as pd
import numpy as np
import os
import time
import pickle
import random
import math
from datetime import datetime
import json

# Set CustomTkinter appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class AnimatedButton(ctk.CTkButton):
    """Custom button with hover animations"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, event):
        self.configure(font=("SF Pro Display", 16, "bold"))

    def on_leave(self, event):
        self.configure(font=("SF Pro Display", 15, "bold"))


class GlowFrame(ctk.CTkFrame):
    """Frame with glowing border effect"""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(border_width=2, border_color="#3b82f6")


class ModernCounterfactualGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CODEC • Constraint-Guided Counterfactuals")
        self.root.geometry("1600x950")

        # Modern color scheme - Cyberpunk inspired
        self.colors = {
            'bg_primary': '#0a0a0a',
            'bg_secondary': '#111111',
            'bg_tertiary': '#1a1a1a',
            'bg_card': '#161616',
            'accent': '#00d4ff',
            'accent_secondary': '#ff006e',
            'accent_hover': '#00a8cc',
            'success': '#00ff88',
            'warning': '#ffaa00',
            'error': '#ff0055',
            'text_primary': '#ffffff',
            'text_secondary': '#b0b0b0',
            'text_dim': '#666666',
            'border': '#2a2a2a',
            'glow': '#00d4ff'
        }

        # Configure root window
        self.root.configure(bg=self.colors['bg_primary'])

        # Initialize variables
        self.dataset_path = tk.StringVar()
        self.constraints_path = tk.StringVar()
        self.num_counterfactuals = tk.StringVar(value="3")
        self.fixed_features = set()
        self.animation_running = False

        # Initialize caches
        self.cached_models = {}
        self.cached_transformers = {}
        self.cached_constraints = {}
        self.initial_point_widgets = {}

        # Initialize queue
        self.computation_queue = queue.Queue()
        self.check_computation_queue()

        # Create main layout
        self.setup_ui()

    def setup_ui(self):
        # Create main container with subtle gradient
        self.main_container = ctk.CTkFrame(self.root, fg_color=self.colors['bg_primary'])
        self.main_container.pack(fill='both', expand=True)

        # Create animated header
        self.create_animated_header()

        # Create content area
        content_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        content_frame.pack(fill='both', expand=True, padx=30, pady=(0, 30))

        # Create sidebar and main content
        self.create_sidebar(content_frame)
        self.create_main_content(content_frame)

    def create_animated_header(self):
        """Create an animated header with gradient and particles"""
        header_frame = ctk.CTkFrame(self.main_container, fg_color=self.colors['bg_secondary'], height=120)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)

        # Canvas for animated background
        self.header_canvas = tk.Canvas(
            header_frame,
            height=120,
            bg=self.colors['bg_secondary'],
            highlightthickness=0
        )
        self.header_canvas.pack(fill='both', expand=True)

        # Title with glow effect
        title_text = self.header_canvas.create_text(
            80, 60,
            text="CODEC",
            font=("SF Pro Display", 48, "bold"),
            fill=self.colors['accent'],
            anchor='w'
        )

        # Subtitle
        subtitle_text = self.header_canvas.create_text(
            80, 90,
            text="Constraint-Optimized Diverse Explanatory Counterfactuals",
            font=("SF Pro Display", 16),
            fill=self.colors['text_secondary'],
            anchor='w'
        )

        # Status indicator
        self.status_frame = ctk.CTkFrame(
            header_frame,
            fg_color=self.colors['bg_tertiary'],
            corner_radius=20,
            width=200,
            height=40
        )
        self.status_frame.place(relx=0.85, rely=0.5, anchor='center')

        self.status_indicator = ctk.CTkLabel(
            self.status_frame,
            text="● Ready",
            font=("SF Pro Display", 14),
            text_color=self.colors['success']
        )
        self.status_indicator.pack(expand=True)

        # Start header animation
        self.animate_header_particles()

    def animate_header_particles(self):
        """Create floating particle animation in header"""
        if not hasattr(self, 'particles'):
            self.particles = []
            for _ in range(15):
                x = random.randint(0, 1600)
                y = random.randint(0, 120)
                size = random.randint(2, 4)
                speed = random.uniform(0.5, 2)
                particle = self.header_canvas.create_oval(
                    x, y, x + size, y + size,
                    fill=self.colors['accent'],
                    outline=""
                )
                self.particles.append({'id': particle, 'speed': speed, 'x': x})

        # Move particles
        for particle in self.particles:
            self.header_canvas.move(particle['id'], particle['speed'], 0)
            coords = self.header_canvas.coords(particle['id'])
            if coords[0] > 1600:
                self.header_canvas.coords(particle['id'], -10, coords[1], -10 + coords[2] - coords[0], coords[3])

        self.root.after(50, self.animate_header_particles)

    def create_sidebar(self, parent):
        """Create modern sidebar navigation"""
        sidebar = ctk.CTkFrame(
            parent,
            fg_color=self.colors['bg_secondary'],
            corner_radius=15,
            width=300
        )
        sidebar.pack(side='left', fill='y', padx=(0, 20))
        sidebar.pack_propagate(False)

        # Navigation title
        nav_title = ctk.CTkLabel(
            sidebar,
            text="Navigation",
            font=("SF Pro Display", 20, "bold"),
            text_color=self.colors['text_primary']
        )
        nav_title.pack(pady=20)

        # Navigation buttons
        self.nav_buttons = []
        nav_items = [
            ("📁", "Input Data", self.show_input_view),
            ("🔒", "Constraints", self.show_constraints_view),
            ("📊", "Results", self.show_results_view),
        ]

        for icon, text, command in nav_items:
            btn_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
            btn_frame.pack(fill='x', padx=20, pady=5)

            btn = ctk.CTkButton(
                btn_frame,
                text=f"{icon}  {text}",
                font=("SF Pro Display", 16),
                height=50,
                fg_color="transparent",
                hover_color=self.colors['bg_tertiary'],
                anchor='w',
                command=command
            )
            btn.pack(fill='x')
            self.nav_buttons.append(btn)

        # Set first button as active
        self.nav_buttons[0].configure(fg_color=self.colors['bg_tertiary'])

        # Quick stats
        self.create_quick_stats(sidebar)

    def create_quick_stats(self, parent):
        """Create quick statistics panel"""
        stats_frame = ctk.CTkFrame(
            parent,
            fg_color=self.colors['bg_tertiary'],
            corner_radius=15
        )
        stats_frame.pack(fill='x', side='bottom', padx=20, pady=20)

        stats_title = ctk.CTkLabel(
            stats_frame,
            text="Quick Stats",
            font=("SF Pro Display", 16, "bold"),
            text_color=self.colors['text_primary']
        )
        stats_title.pack(pady=10)

        self.stats_labels = {}
        stats = [
            ("Dataset", "Not loaded"),
            ("Constraints", "Not loaded"),
            ("Model", "Not trained")
        ]

        for stat_name, stat_value in stats:
            stat_frame = ctk.CTkFrame(stats_frame, fg_color="transparent")
            stat_frame.pack(fill='x', padx=15, pady=5)

            name_label = ctk.CTkLabel(
                stat_frame,
                text=f"{stat_name}:",
                font=("SF Pro Display", 12),
                text_color=self.colors['text_secondary']
            )
            name_label.pack(side='left')

            value_label = ctk.CTkLabel(
                stat_frame,
                text=stat_value,
                font=("SF Pro Display", 12, "bold"),
                text_color=self.colors['accent']
            )
            value_label.pack(side='right')

            self.stats_labels[stat_name] = value_label

    def create_main_content(self, parent):
        """Create main content area with views"""
        self.content_frame = ctk.CTkFrame(
            parent,
            fg_color=self.colors['bg_secondary'],
            corner_radius=15
        )
        self.content_frame.pack(side='right', fill='both', expand=True)

        # Create different views
        self.views = {}
        self.create_input_view()
        self.create_constraints_view()
        self.create_results_view()

        # Show input view by default
        self.show_input_view()

    def create_input_view(self):
        """Create modern input configuration view"""
        view = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.views['input'] = view

        # Scrollable frame
        scroll_frame = ctk.CTkScrollableFrame(
            view,
            fg_color="transparent",
            scrollbar_button_color=self.colors['bg_tertiary'],
            scrollbar_button_hover_color=self.colors['accent']
        )
        scroll_frame.pack(fill='both', expand=True, padx=30, pady=30)

        # Dataset section
        dataset_card = self.create_card(scroll_frame, "Dataset Configuration")

        # Dataset path input
        dataset_frame = ctk.CTkFrame(dataset_card, fg_color="transparent")
        dataset_frame.pack(fill='x', pady=10)

        ctk.CTkLabel(
            dataset_frame,
            text="Dataset File",
            font=("SF Pro Display", 14),
            text_color=self.colors['text_secondary']
        ).pack(anchor='w')

        input_frame = ctk.CTkFrame(dataset_frame, fg_color="transparent")
        input_frame.pack(fill='x', pady=5)

        self.dataset_entry = ctk.CTkEntry(
            input_frame,
            textvariable=self.dataset_path,
            height=40,
            font=("SF Pro Display", 14),
            fg_color=self.colors['bg_primary'],
            border_color=self.colors['border'],
            border_width=2
        )
        self.dataset_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))

        AnimatedButton(
            input_frame,
            text="Browse",
            width=100,
            height=40,
            font=("SF Pro Display", 15, "bold"),
            fg_color=self.colors['accent'],
            hover_color=self.colors['accent_hover'],
            command=self.browse_dataset
        ).pack(side='right')

        # Constraints section
        constraints_frame = ctk.CTkFrame(dataset_card, fg_color="transparent")
        constraints_frame.pack(fill='x', pady=10)

        ctk.CTkLabel(
            constraints_frame,
            text="Constraints File",
            font=("SF Pro Display", 14),
            text_color=self.colors['text_secondary']
        ).pack(anchor='w')

        input_frame2 = ctk.CTkFrame(constraints_frame, fg_color="transparent")
        input_frame2.pack(fill='x', pady=5)

        self.constraints_entry = ctk.CTkEntry(
            input_frame2,
            textvariable=self.constraints_path,
            height=40,
            font=("SF Pro Display", 14),
            fg_color=self.colors['bg_primary'],
            border_color=self.colors['border'],
            border_width=2
        )
        self.constraints_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))

        AnimatedButton(
            input_frame2,
            text="Browse",
            width=100,
            height=40,
            font=("SF Pro Display", 15, "bold"),
            fg_color=self.colors['accent'],
            hover_color=self.colors['accent_hover'],
            command=self.browse_constraints
        ).pack(side='right')

        # Parameters section
        params_card = self.create_card(scroll_frame, "Generation Parameters")

        # Number of counterfactuals
        num_cf_frame = ctk.CTkFrame(params_card, fg_color="transparent")
        num_cf_frame.pack(fill='x', pady=10)

        ctk.CTkLabel(
            num_cf_frame,
            text="Number of Counterfactuals",
            font=("SF Pro Display", 14),
            text_color=self.colors['text_secondary']
        ).pack(side='left')

        self.num_cf_slider = ctk.CTkSlider(
            num_cf_frame,
            from_=1,
            to=10,
            number_of_steps=9,
            height=20,
            progress_color=self.colors['accent'],
            button_color=self.colors['accent'],
            button_hover_color=self.colors['accent_hover'],
            command=self.update_cf_count
        )
        self.num_cf_slider.pack(side='left', fill='x', expand=True, padx=20)
        self.num_cf_slider.set(3)

        self.cf_count_label = ctk.CTkLabel(
            num_cf_frame,
            text="3",
            font=("SF Pro Display", 18, "bold"),
            text_color=self.colors['accent'],
            width=30
        )
        self.cf_count_label.pack(side='right')

        # Initial instance section
        self.instance_card = self.create_card(scroll_frame, "Initial Instance Configuration")
        self.instance_content_frame = ctk.CTkFrame(self.instance_card, fg_color="transparent")
        self.instance_content_frame.pack(fill='both', expand=True)

        # Placeholder for initial instance
        placeholder_label = ctk.CTkLabel(
            self.instance_content_frame,
            text="Load a dataset to configure initial instance",
            font=("SF Pro Display", 14),
            text_color=self.colors['text_dim']
        )
        placeholder_label.pack(pady=40)

        # Fixed features section
        self.features_card = self.create_card(scroll_frame, "Immutable Features")
        self.features_content_frame = ctk.CTkFrame(self.features_card, fg_color="transparent")
        self.features_content_frame.pack(fill='both', expand=True)

        # Placeholder for features
        features_placeholder = ctk.CTkLabel(
            self.features_content_frame,
            text="Load a dataset to select immutable features",
            font=("SF Pro Display", 14),
            text_color=self.colors['text_dim']
        )
        features_placeholder.pack(pady=40)

        # Generate button
        self.generate_btn = AnimatedButton(
            scroll_frame,
            text="🚀 Generate Counterfactuals",
            height=60,
            font=("SF Pro Display", 18, "bold"),
            fg_color=self.colors['accent'],
            hover_color=self.colors['accent_hover'],
            command=self.generate_counterfactuals
        )
        self.generate_btn.pack(fill='x', pady=30)

    def create_constraints_view(self):
        """Create modern constraints visualization view"""
        view = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.views['constraints'] = view

        # Header
        header_frame = ctk.CTkFrame(view, fg_color="transparent")
        header_frame.pack(fill='x', padx=30, pady=(30, 20))

        ctk.CTkLabel(
            header_frame,
            text="Active Denial Constraints",
            font=("SF Pro Display", 28, "bold"),
            text_color=self.colors['text_primary']
        ).pack(side='left')

        # Constraints display
        self.constraints_display = ctk.CTkTextbox(
            view,
            font=("SF Mono", 14),
            fg_color=self.colors['bg_primary'],
            text_color=self.colors['text_primary'],
            scrollbar_button_color=self.colors['bg_tertiary'],
            scrollbar_button_hover_color=self.colors['accent'],
            corner_radius=15,
            border_width=2,
            border_color=self.colors['border']
        )
        self.constraints_display.pack(fill='both', expand=True, padx=30, pady=(0, 30))

    def create_results_view(self):
        """Create modern results visualization view"""
        view = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.views['results'] = view

        # Header with metrics
        header_frame = ctk.CTkFrame(view, fg_color="transparent")
        header_frame.pack(fill='x', padx=30, pady=(30, 20))

        ctk.CTkLabel(
            header_frame,
            text="Counterfactual Results",
            font=("SF Pro Display", 28, "bold"),
            text_color=self.colors['text_primary']
        ).pack(side='left')

        # Metrics cards
        metrics_frame = ctk.CTkFrame(view, fg_color="transparent")
        metrics_frame.pack(fill='x', padx=30, pady=(0, 20))

        # Diversity score card
        diversity_card = ctk.CTkFrame(
            metrics_frame,
            fg_color=self.colors['bg_tertiary'],
            corner_radius=15,
            height=100
        )
        diversity_card.pack(side='left', fill='x', expand=True, padx=(0, 10))
        diversity_card.pack_propagate(False)

        ctk.CTkLabel(
            diversity_card,
            text="Diversity Score",
            font=("SF Pro Display", 14),
            text_color=self.colors['text_secondary']
        ).pack(pady=(15, 5))

        self.diversity_label = ctk.CTkLabel(
            diversity_card,
            text="--",
            font=("SF Pro Display", 32, "bold"),
            text_color=self.colors['accent']
        )
        self.diversity_label.pack()

        # Distance score card
        distance_card = ctk.CTkFrame(
            metrics_frame,
            fg_color=self.colors['bg_tertiary'],
            corner_radius=15,
            height=100
        )
        distance_card.pack(side='left', fill='x', expand=True, padx=(10, 0))
        distance_card.pack_propagate(False)

        ctk.CTkLabel(
            distance_card,
            text="Avg. Distance",
            font=("SF Pro Display", 14),
            text_color=self.colors['text_secondary']
        ).pack(pady=(15, 5))

        self.distance_label = ctk.CTkLabel(
            distance_card,
            text="--",
            font=("SF Pro Display", 32, "bold"),
            text_color=self.colors['success']
        )
        self.distance_label.pack()

        # Results display
        self.results_frame = ctk.CTkScrollableFrame(
            view,
            fg_color=self.colors['bg_primary'],
            scrollbar_button_color=self.colors['bg_tertiary'],
            scrollbar_button_hover_color=self.colors['accent'],
            corner_radius=15
        )
        self.results_frame.pack(fill='both', expand=True, padx=30, pady=(0, 30))

        # Placeholder
        self.results_placeholder = ctk.CTkLabel(
            self.results_frame,
            text="Generate counterfactuals to see results",
            font=("SF Pro Display", 16),
            text_color=self.colors['text_dim']
        )
        self.results_placeholder.pack(pady=100)

    def create_card(self, parent, title):
        """Create a modern card component"""
        card = ctk.CTkFrame(
            parent,
            fg_color=self.colors['bg_tertiary'],
            corner_radius=15
        )
        card.pack(fill='x', pady=10)

        # Card header
        header = ctk.CTkFrame(card, fg_color="transparent")
        header.pack(fill='x', padx=20, pady=(20, 10))

        ctk.CTkLabel(
            header,
            text=title,
            font=("SF Pro Display", 18, "bold"),
            text_color=self.colors['text_primary']
        ).pack(side='left')

        # Card content
        content = ctk.CTkFrame(card, fg_color="transparent")
        content.pack(fill='both', expand=True, padx=20, pady=(0, 20))

        return content

    def update_cf_count(self, value):
        """Update counterfactual count label"""
        self.cf_count_label.configure(text=str(int(value)))
        self.num_counterfactuals.set(str(int(value)))

    def show_input_view(self):
        """Show input configuration view"""
        self.switch_view('input', 0)

    def show_constraints_view(self):
        """Show constraints view"""
        self.switch_view('constraints', 1)

    def show_results_view(self):
        """Show results view"""
        self.switch_view('results', 2)

    def switch_view(self, view_name, button_index):
        """Switch between different views"""
        # Hide all views
        for view in self.views.values():
            view.pack_forget()

        # Show selected view
        self.views[view_name].pack(fill='both', expand=True)

        # Update navigation buttons
        for i, btn in enumerate(self.nav_buttons):
            if i == button_index:
                btn.configure(fg_color=self.colors['bg_tertiary'])
            else:
                btn.configure(fg_color="transparent")

    def browse_dataset(self):
        """Browse for dataset file"""
        filename = filedialog.askopenfilename(
            title="Select Dataset CSV",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if filename:
            self.dataset_path.set(filename)
            self.stats_labels["Dataset"].configure(text="✓ Loaded")
            self.update_status("Dataset loaded", self.colors['success'])
            self.load_dataset_preview()

    def browse_constraints(self):
        """Browse for constraints file"""
        filename = filedialog.askopenfilename(
            title="Select Constraints File",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
        )
        if filename:
            self.constraints_path.set(filename)
            self.stats_labels["Constraints"].configure(text="✓ Loaded")
            self.update_status("Constraints loaded", self.colors['success'])
            self.load_constraints_preview()

    def update_status(self, text, color):
        """Update status indicator"""
        self.status_indicator.configure(text=f"● {text}", text_color=color)

    def load_dataset_preview(self):
        """Load and display dataset preview"""
        try:
            # Load dataset
            self.df = pd.read_csv(self.dataset_path.get())

            # Update initial instance configuration
            self.setup_initial_instance_inputs()

            # Update features list
            self.setup_features_selection()

        except Exception as e:
            self.update_status(f"Error: {str(e)}", self.colors['error'])

    def setup_initial_instance_inputs(self):
        """Setup initial instance configuration inputs"""
        # Clear existing widgets
        for widget in self.instance_content_frame.winfo_children():
            widget.destroy()

        # Create inputs for each feature
        for i, column in enumerate(self.df.columns):
            if column == 'label':
                continue

            # Feature frame
            feature_frame = ctk.CTkFrame(self.instance_content_frame, fg_color="transparent")
            feature_frame.pack(fill='x', pady=5)

            # Feature label
            ctk.CTkLabel(
                feature_frame,
                text=column,
                font=("SF Pro Display", 14),
                text_color=self.colors['text_secondary'],
                width=150,
                anchor='w'
            ).pack(side='left')

            if self.df[column].dtype == 'object':  # Categorical
                unique_values = sorted(self.df[column].unique())
                widget = ctk.CTkComboBox(
                    feature_frame,
                    values=unique_values,
                    font=("SF Pro Display", 14),
                    fg_color=self.colors['bg_primary'],
                    button_color=self.colors['accent'],
                    button_hover_color=self.colors['accent_hover'],
                    dropdown_fg_color=self.colors['bg_primary'],
                    dropdown_hover_color=self.colors['bg_tertiary'],
                    width=200
                )
                widget.set(unique_values[0])
            else:  # Continuous
                min_val = self.df[column].min()
                max_val = self.df[column].max()
                widget = ctk.CTkEntry(
                    feature_frame,
                    font=("SF Pro Display", 14),
                    fg_color=self.colors['bg_primary'],
                    border_color=self.colors['border'],
                    border_width=2,
                    placeholder_text=f"{min_val} - {max_val}",
                    width=200
                )
                widget.insert(0, str(int(min_val)))

            widget.pack(side='right', padx=10)
            self.initial_point_widgets[column] = widget

    def setup_features_selection(self):
        """Setup immutable features selection"""
        # Clear existing widgets
        for widget in self.features_content_frame.winfo_children():
            widget.destroy()

        # Create checkbox for each feature
        self.feature_checkboxes = {}

        for column in self.df.columns:
            if column == 'label':
                continue

            var = tk.BooleanVar(value=False)

            checkbox = ctk.CTkCheckBox(
                self.features_content_frame,
                text=column,
                font=("SF Pro Display", 14),
                variable=var,
                fg_color=self.colors['accent'],
                hover_color=self.colors['accent_hover'],
                border_color=self.colors['border'],
                checkmark_color=self.colors['bg_primary']
            )
            checkbox.pack(anchor='w', pady=5)

            self.feature_checkboxes[column] = var

    def load_constraints_preview(self):
        """Load and display constraints"""
        try:
            with open(self.constraints_path.get(), 'r', encoding='utf-8') as file:
                constraints = file.read()

            self.constraints_display.delete("1.0", "end")
            self.constraints_display.insert("1.0", constraints)

        except Exception as e:
            self.update_status(f"Error: {str(e)}", self.colors['error'])

    def get_initial_point(self):
        """Get initial point values"""
        if not hasattr(self, 'initial_point_widgets') or not self.initial_point_widgets:
            return None

        initial_point = {'label': 0}

        for column, widget in self.initial_point_widgets.items():
            try:
                if isinstance(widget, ctk.CTkComboBox):
                    initial_point[column] = widget.get()
                else:
                    value = widget.get()
                    try:
                        initial_point[column] = float(value)
                    except ValueError:
                        initial_point[column] = value
            except:
                return None

        return initial_point

    def get_fixed_features(self):
        """Get selected immutable features"""
        fixed = []
        for column, var in self.feature_checkboxes.items():
            if var.get():
                fixed.append(column)
        return fixed

    def generate_counterfactuals(self):
        """Generate counterfactuals with modern loading animation"""
        # Validate inputs
        initial_instance = self.get_initial_point()
        if not initial_instance:
            self.update_status("Invalid initial instance", self.colors['error'])
            return

        if not self.constraints_path.get() or not self.dataset_path.get():
            self.update_status("Missing required files", self.colors['error'])
            return

        # Update status
        self.update_status("Generating...", self.colors['warning'])

        # Show loading overlay
        self.show_loading_overlay()

        # Start computation in separate thread
        computation_thread = threading.Thread(
            target=self.computation_worker,
            args=(
                initial_instance,
                self.constraints_path.get(),
                self.dataset_path.get(),
                self.get_fixed_features(),
                int(self.num_counterfactuals.get())
            )
        )
        computation_thread.daemon = True
        computation_thread.start()

    def show_loading_overlay(self):
        """Show modern loading overlay"""
        self.loading_overlay = ctk.CTkFrame(
            self.root,
            fg_color=(self.colors['bg_primary'], self.colors['bg_primary']),
            corner_radius=0
        )
        self.loading_overlay.place(relx=0.5, rely=0.5, anchor='center', relwidth=1, relheight=1)

        # Loading content
        loading_content = ctk.CTkFrame(
            self.loading_overlay,
            fg_color=self.colors['bg_secondary'],
            corner_radius=20,
            width=400,
            height=300
        )
        loading_content.place(relx=0.5, rely=0.5, anchor='center')
        loading_content.pack_propagate(False)

        # Loading animation
        self.loading_animation_frame = ctk.CTkFrame(
            loading_content,
            fg_color="transparent",
            height=100
        )
        self.loading_animation_frame.pack(pady=(40, 20))

        # Create spinning circles
        self.create_loading_animation()

        # Loading text
        self.loading_text = ctk.CTkLabel(
            loading_content,
            text="Initializing CODEC...",
            font=("SF Pro Display", 18, "bold"),
            text_color=self.colors['text_primary']
        )
        self.loading_text.pack(pady=10)

        # Progress text
        self.progress_text = ctk.CTkLabel(
            loading_content,
            text="",
            font=("SF Pro Display", 14),
            text_color=self.colors['text_secondary']
        )
        self.progress_text.pack(pady=5)

    def create_loading_animation(self):
        """Create modern loading animation"""
        canvas = tk.Canvas(
            self.loading_animation_frame,
            width=100,
            height=100,
            bg=self.colors['bg_secondary'],
            highlightthickness=0
        )
        canvas.pack()

        # Create multiple circles
        self.loading_circles = []
        for i in range(8):
            angle = i * 45
            x = 50 + 30 * math.cos(math.radians(angle))
            y = 50 + 30 * math.sin(math.radians(angle))

            circle = canvas.create_oval(
                x - 5, y - 5, x + 5, y + 5,
                fill=self.colors['accent'],
                outline=""
            )
            self.loading_circles.append(circle)

        self.loading_canvas = canvas
        self.animate_loading()

    def animate_loading(self, index=0):
        """Animate loading circles"""
        if hasattr(self, 'loading_canvas'):
            for i, circle in enumerate(self.loading_circles):
                # Calculate opacity based on position
                opacity_index = (index - i) % len(self.loading_circles)
                opacity = 1.0 - (opacity_index / len(self.loading_circles))

                # Update circle size based on opacity
                size = 5 + opacity * 3
                coords = self.loading_canvas.coords(circle)
                center_x = (coords[0] + coords[2]) / 2
                center_y = (coords[1] + coords[3]) / 2

                self.loading_canvas.coords(
                    circle,
                    center_x - size, center_y - size,
                    center_x + size, center_y + size
                )

            self.root.after(100, lambda: self.animate_loading((index + 1) % len(self.loading_circles)))

    def computation_worker(self, initial_instance, constraints_path, dataset_path, fixed_features, num_counterfactuals):
        """Worker function for computation thread"""
        try:
            # Import the actual computation function here
            # For demo purposes, we'll simulate the computation

            # Simulate progress updates
            progress_messages = [
                "Loading dataset and constraints...",
                "Initializing DiCE framework...",
                "Training model...",
                "Generating initial counterfactuals...",
                "Applying constraint projections...",
                "Optimizing solutions...",
                "Finalizing results..."
            ]

            for i, msg in enumerate(progress_messages):
                self.computation_queue.put({'type': 'progress', 'text': msg})
                time.sleep(1)  # Simulate work

            # Simulate results
            # In real implementation, call your code_counterfactuals function here
            results = self.generate_mock_results(initial_instance, num_counterfactuals)

            self.computation_queue.put({
                'type': 'complete',
                'data': results
            })

        except Exception as e:
            self.computation_queue.put({'type': 'error', 'error': str(e)})

    def generate_mock_results(self, initial_instance, num_counterfactuals):
        """Generate mock results for demonstration"""
        # Create mock counterfactuals
        counterfactuals = []

        for i in range(num_counterfactuals):
            cf = initial_instance.copy()
            # Modify some values
            for key in cf:
                if key != 'label' and random.random() < 0.3:
                    if isinstance(cf[key], (int, float)):
                        cf[key] = cf[key] * random.uniform(0.8, 1.2)
                    elif isinstance(cf[key], str) and hasattr(self, 'df'):
                        col_values = self.df[key].unique()
                        if len(col_values) > 1:
                            cf[key] = random.choice([v for v in col_values if v != cf[key]])
            counterfactuals.append(cf)

        return {
            'initial_instance': initial_instance,
            'counterfactuals': counterfactuals,
            'diversity_score': random.uniform(0.7, 0.9),
            'distances': [random.uniform(0.1, 0.5) for _ in range(num_counterfactuals)]
        }

    def check_computation_queue(self):
        """Check for messages from computation thread"""
        try:
            while True:
                message = self.computation_queue.get_nowait()
                if message['type'] == 'progress':
                    self.update_progress(message['text'])
                elif message['type'] == 'complete':
                    self.handle_computation_complete(message['data'])
                elif message['type'] == 'error':
                    self.handle_computation_error(message['error'])
        except queue.Empty:
            pass

        self.root.after(50, self.check_computation_queue)

    def update_progress(self, text):
        """Update progress text during computation"""
        if hasattr(self, 'progress_text'):
            self.progress_text.configure(text=text)

    def handle_computation_complete(self, data):
        """Handle successful computation"""
        # Remove loading overlay
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.destroy()

        # Update status
        self.update_status("Generation complete", self.colors['success'])
        self.stats_labels["Model"].configure(text="✓ Trained")

        # Display results
        self.display_results(data)

        # Switch to results view
        self.show_results_view()

    def handle_computation_error(self, error):
        """Handle computation error"""
        # Remove loading overlay
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.destroy()

        # Update status
        self.update_status(f"Error: {error}", self.colors['error'])

        # Show error message
        messagebox.showerror("Computation Error", f"Failed to generate counterfactuals:\n{error}")

    def display_results(self, data):
        """Display results in modern format"""
        # Clear existing results
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        # Update metrics
        self.diversity_label.configure(text=f"{data['diversity_score']:.3f}")
        avg_distance = sum(data['distances']) / len(data['distances'])
        self.distance_label.configure(text=f"{avg_distance:.3f}")

        # Display initial instance
        initial_card = self.create_result_card(
            self.results_frame,
            "Initial Instance",
            data['initial_instance'],
            is_initial=True
        )

        # Display counterfactuals
        for i, cf in enumerate(data['counterfactuals']):
            distance = data['distances'][i]
            self.create_result_card(
                self.results_frame,
                f"Counterfactual Option {i + 1}",
                cf,
                is_initial=False,
                distance=distance,
                initial_instance=data['initial_instance']
            )

    def create_result_card(self, parent, title, instance, is_initial=False, distance=None, initial_instance=None):
        """Create a modern result card"""
        # Card container
        card = ctk.CTkFrame(
            parent,
            fg_color=self.colors['bg_tertiary'] if not is_initial else self.colors['bg_secondary'],
            corner_radius=15
        )
        card.pack(fill='x', pady=10, padx=20)

        # Card header
        header_frame = ctk.CTkFrame(card, fg_color="transparent")
        header_frame.pack(fill='x', padx=20, pady=(20, 10))

        # Title
        title_label = ctk.CTkLabel(
            header_frame,
            text=title,
            font=("SF Pro Display", 18, "bold"),
            text_color=self.colors['text_primary']
        )
        title_label.pack(side='left')

        # Distance badge
        if distance is not None:
            distance_badge = ctk.CTkFrame(
                header_frame,
                fg_color=self.colors['accent'],
                corner_radius=10
            )
            distance_badge.pack(side='right')

            ctk.CTkLabel(
                distance_badge,
                text=f"Distance: {distance:.3f}",
                font=("SF Pro Display", 12),
                text_color=self.colors['bg_primary']
            ).pack(padx=10, pady=5)

        # Attributes grid
        attributes_frame = ctk.CTkFrame(card, fg_color="transparent")
        attributes_frame.pack(fill='x', padx=20, pady=(0, 20))

        # Create grid of attributes
        row = 0
        col = 0
        max_cols = 3

        for key, value in instance.items():
            if key == 'label':
                continue

            # Attribute frame
            attr_frame = ctk.CTkFrame(
                attributes_frame,
                fg_color=self.colors['bg_primary'] if not is_initial else self.colors['bg_tertiary'],
                corner_radius=10,
                width=350
            )
            attr_frame.grid(row=row, column=col, padx=5, pady=5, sticky='ew')
            attr_frame.pack_propagate(False)

            # Attribute content
            content_frame = ctk.CTkFrame(attr_frame, fg_color="transparent")
            content_frame.pack(fill='x', padx=15, pady=10)

            # Attribute name
            ctk.CTkLabel(
                content_frame,
                text=key,
                font=("SF Pro Display", 12),
                text_color=self.colors['text_secondary']
            ).pack(side='left')

            # Check if value changed
            changed = False
            if initial_instance and key in initial_instance:
                if isinstance(value, (int, float)):
                    changed = abs(value - initial_instance[key]) > 0.001
                else:
                    changed = str(value) != str(initial_instance[key])

            # Attribute value
            value_text = f"{value:.2f}" if isinstance(value, float) else str(value)
            if changed and not is_initial:
                value_text = f"→ {value_text}"

            value_label = ctk.CTkLabel(
                content_frame,
                text=value_text,
                font=("SF Pro Display", 13, "bold"),
                text_color=self.colors['accent'] if changed else self.colors['text_primary']
            )
            value_label.pack(side='right')

            # Update grid position
            col += 1
            if col >= max_cols:
                col = 0
                row += 1

        # Configure grid weights
        for i in range(max_cols):
            attributes_frame.columnconfigure(i, weight=1)

        return card


def main():
    root = ctk.CTk()
    app = ModernCounterfactualGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()