/**
 * Animation Utilities and Variants
 * Reusable Framer Motion animation presets
 */

import { Variants } from 'framer-motion';

/**
 * Fade In Animation
 */
export const fadeIn: Variants = {
  hidden: { opacity: 0 },
  visible: { 
    opacity: 1,
    transition: { duration: 0.3 }
  },
  exit: { 
    opacity: 0,
    transition: { duration: 0.2 }
  }
};

/**
 * Slide Up Animation
 */
export const slideUp: Variants = {
  hidden: { opacity: 0, y: 20 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: { duration: 0.4, ease: 'easeOut' }
  },
  exit: { 
    opacity: 0, 
    y: -20,
    transition: { duration: 0.3 }
  }
};

/**
 * Slide Down Animation
 */
export const slideDown: Variants = {
  hidden: { opacity: 0, y: -20 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: { duration: 0.4, ease: 'easeOut' }
  },
  exit: { 
    opacity: 0, 
    y: 20,
    transition: { duration: 0.3 }
  }
};

/**
 * Slide In From Left
 */
export const slideInLeft: Variants = {
  hidden: { opacity: 0, x: -50 },
  visible: { 
    opacity: 1, 
    x: 0,
    transition: { duration: 0.4, ease: 'easeOut' }
  },
  exit: { 
    opacity: 0, 
    x: -50,
    transition: { duration: 0.3 }
  }
};

/**
 * Slide In From Right
 */
export const slideInRight: Variants = {
  hidden: { opacity: 0, x: 50 },
  visible: { 
    opacity: 1, 
    x: 0,
    transition: { duration: 0.4, ease: 'easeOut' }
  },
  exit: { 
    opacity: 0, 
    x: 50,
    transition: { duration: 0.3 }
  }
};

/**
 * Scale In Animation
 */
export const scaleIn: Variants = {
  hidden: { opacity: 0, scale: 0.8 },
  visible: { 
    opacity: 1, 
    scale: 1,
    transition: { duration: 0.3, ease: 'easeOut' }
  },
  exit: { 
    opacity: 0, 
    scale: 0.8,
    transition: { duration: 0.2 }
  }
};

/**
 * Stagger Children Animation
 */
export const staggerContainer: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.1
    }
  }
};

export const staggerItem: Variants = {
  hidden: { opacity: 0, y: 20 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: { duration: 0.4 }
  }
};

/**
 * Bounce Animation
 */
export const bounce: Variants = {
  hidden: { opacity: 0, scale: 0 },
  visible: { 
    opacity: 1, 
    scale: 1,
    transition: { 
      type: 'spring',
      stiffness: 260,
      damping: 20 
    }
  }
};

/**
 * Rotate In Animation
 */
export const rotateIn: Variants = {
  hidden: { opacity: 0, rotate: -180 },
  visible: { 
    opacity: 1, 
    rotate: 0,
    transition: { duration: 0.5, ease: 'easeOut' }
  }
};

/**
 * Flip Animation
 */
export const flip: Variants = {
  hidden: { opacity: 0, rotateY: -90 },
  visible: { 
    opacity: 1, 
    rotateY: 0,
    transition: { duration: 0.6, ease: 'easeOut' }
  }
};

/**
 * Modal Animation
 */
export const modal: Variants = {
  hidden: { 
    opacity: 0, 
    scale: 0.95,
    y: 20
  },
  visible: { 
    opacity: 1, 
    scale: 1,
    y: 0,
    transition: { 
      duration: 0.3,
      ease: 'easeOut'
    }
  },
  exit: { 
    opacity: 0, 
    scale: 0.95,
    y: 20,
    transition: { 
      duration: 0.2 
    }
  }
};

/**
 * Backdrop Animation
 */
export const backdrop: Variants = {
  hidden: { opacity: 0 },
  visible: { 
    opacity: 1,
    transition: { duration: 0.2 }
  },
  exit: { 
    opacity: 0,
    transition: { duration: 0.2 }
  }
};

/**
 * Progress Bar Animation
 */
export const progressBar = {
  initial: { scaleX: 0, originX: 0 },
  animate: (percentage: number) => ({
    scaleX: percentage / 100,
    transition: { duration: 0.5, ease: 'easeOut' }
  })
};

/**
 * Pulse Animation (for attention)
 */
export const pulse = {
  scale: [1, 1.05, 1],
  transition: {
    duration: 2,
    repeat: Infinity,
    ease: 'easeInOut'
  }
};

/**
 * Shake Animation (for errors)
 */
export const shake = {
  x: [0, -10, 10, -10, 10, 0],
  transition: {
    duration: 0.5
  }
};

/**
 * Hover Lift Effect
 */
export const hoverLift = {
  rest: { y: 0 },
  hover: { 
    y: -8,
    transition: {
      duration: 0.2,
      ease: 'easeOut'
    }
  }
};

/**
 * Tap Scale Effect
 */
export const tapScale = {
  whileTap: { scale: 0.95 }
};

/**
 * Loading Dots Animation
 */
export const loadingDots = (index: number) => ({
  y: [0, -10, 0],
  transition: {
    duration: 0.6,
    repeat: Infinity,
    delay: index * 0.1
  }
});

/**
 * Gradient Animation
 */
export const gradientShift = {
  animate: {
    backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'],
    transition: {
      duration: 3,
      repeat: Infinity,
      ease: 'linear'
    }
  }
};

/**
 * Typewriter Effect Config
 */
export const typewriter = (text: string, delay = 0.05) => ({
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: delay,
      delayChildren: 0.2
    }
  }
});

export const typewriterChar: Variants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1 }
};

/**
 * Page Transition
 */
export const pageTransition: Variants = {
  initial: { opacity: 0, x: -20 },
  animate: { 
    opacity: 1, 
    x: 0,
    transition: { duration: 0.4, ease: 'easeOut' }
  },
  exit: { 
    opacity: 0, 
    x: 20,
    transition: { duration: 0.3 }
  }
};

/**
 * Notification Toast Animation
 */
export const toast: Variants = {
  hidden: { 
    opacity: 0, 
    y: -50,
    scale: 0.3 
  },
  visible: { 
    opacity: 1, 
    y: 0,
    scale: 1,
    transition: {
      type: 'spring',
      stiffness: 300,
      damping: 20
    }
  },
  exit: { 
    opacity: 0,
    scale: 0.5,
    transition: { duration: 0.2 }
  }
};
