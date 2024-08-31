#include <linux/delay.h>
#include <linux/gpio/consumer.h>
#include <linux/interrupt.h>
#include <linux/irq.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/of_gpio.h>
#include <linux/pid.h>
#include <linux/proc_fs.h>
#include <linux/sched.h>
#include <linux/signal.h>

#define PROC_NAME "gpio_interrupt_pid"
#define GPIO_PIN 588

static unsigned int irq_num;
static struct task_struct *user_task = NULL;
static struct proc_dir_entry *proc_file;
static struct gpio_desc *gpiod;

static irqreturn_t gpio_isr(int irq, void *dev_id) {
    struct kernel_siginfo info;
    int ret;

    if (user_task == NULL || !pid_alive(user_task)) {
        printk(KERN_ERR "User-space process not registered\n");
        return IRQ_HANDLED;
    }

    memset(&info, 0, sizeof(struct kernel_siginfo));
    info.si_signo = SIGUSR1;
    info.si_code = SI_QUEUE;
    info.si_int = 1; // Unused

    ret = send_sig_info(SIGUSR1, &info, user_task);
    if (ret < 0) {
        printk(KERN_ERR "Failed to send signal to user-space process\n");
    } else {
        printk(KERN_INFO "Signal sent to user-space process\n");
    }

    return IRQ_HANDLED;
}

static ssize_t register_user_pid(struct file *file, const char __user *buffer, size_t count, loff_t *pos) {
    pid_t pid;
    struct pid *pid_struct;

    if (kstrtoint_from_user(buffer, count, 10, &pid) != 0) {
        printk(KERN_ERR "Failed to convert user-space PID\n");
        return -EFAULT;
    }

    if (user_task != NULL && pid_alive(user_task)) {
        printk(KERN_ERR "User-space process already registered\n");
        return -EEXIST;
    }

    pid_struct = find_get_pid(pid);
    user_task = pid_task(pid_struct, PIDTYPE_PID);

    if (user_task == NULL) {
        printk(KERN_ERR "Failed to find user-space process with PID %d\n", pid);
        return -ESRCH;
    }

    printk(KERN_INFO "User-space process registered with PID %d\n", pid);
    return count;
}

static const struct proc_ops proc_fops = {
    .proc_write = register_user_pid,
};

static int __init gpio_isr_init(void) {
    int result = 0;

    gpiod = gpio_to_desc(GPIO_PIN);
    if (IS_ERR(gpiod)) {
        printk(KERN_ERR "Failed to get GPIO descriptor for GPIO %d, error: %ld\n", GPIO_PIN, PTR_ERR(gpiod));
        return PTR_ERR(gpiod);
    }

    result = gpiod_direction_input(gpiod);
    if (result < 0) {
        printk(KERN_ERR "Failed to set GPIO %d as input, error: %d\n", GPIO_PIN, result);
        goto error_direction;
    }

    irq_num = gpiod_to_irq(gpiod);
    if (irq_num < 0) {
        printk(KERN_ERR "gpiod_to_irq failed for GPIO %d, error: %d\n", GPIO_PIN, irq_num);
        result = irq_num;
        goto error_irq;
    }

    result = request_irq(irq_num,
                         (irq_handler_t) gpio_isr,
                         IRQF_TRIGGER_RISING,
                         "gpio_interrupt",
                         gpiod);
    if (result < 0) {
        printk(KERN_ERR "GPIO IRQ request failed for GPIO %d, IRQ %d, error: %d\n", GPIO_PIN, irq_num, result);
        goto error_request_irq;
    }

    proc_file = proc_create(PROC_NAME, 0222, NULL, &proc_fops);
    if (!proc_file) {
        printk(KERN_ERR "Failed to create proc file\n");
        result = -ENOMEM;
        goto error_proc_create;
    }

    printk(KERN_INFO "gpio_interrupt_driver loaded successfully for GPIO %d\n", GPIO_PIN);
    return 0;

error_proc_create:
    free_irq(irq_num, gpiod);
error_request_irq:
error_irq:
error_direction:
    gpiod_put(gpiod);
    return result;
}

static void cleanup_user_task(void) {
    if (user_task) {
        put_task_struct(user_task);
        user_task = NULL;
    }
}

static void __exit gpio_isr_exit(void) {
    cleanup_user_task();
    if (irq_num > 0) {
        free_irq(irq_num, gpiod);
    }
    if (gpiod) {
        gpiod_put(gpiod);
    }
    if (proc_file) {
        proc_remove(proc_file);
    }
    printk(KERN_INFO "gpio_interrupt_driver unloaded for GPIO %d\n", GPIO_PIN);
}

module_init(gpio_isr_init);
module_exit(gpio_isr_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("GPIO 17 interrupt with user-space signal");
MODULE_AUTHOR("Alec Fessler");
