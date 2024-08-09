#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/gpio.h>
#include <linux/interrupt.h>
#include <linux/irq.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/signal.h>

static unsigned int gpio_pin = 17;
static unsigned int irq_num;

static struct task_struct *user_task = NULL;

static irqreturn_t gpio_isr(int irq, void *dev_id) {
    struct siginfo info;
    int ret;

    if (user_task == NULL) {
        printk(KERN_ERR "User-space process not registered\n");
        return IRQ_HANDLED;
    }

    memset(&info, 0, sizeof(struct siginfo));
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

static int __init gpio_isr_init(void) {
    int result = 0;

    if (!gpio_is_valid(gpio_pin)) {
        printk(KERN_ERR "Invalid GPIO pin\n");
        return -ENODEV;
    }

    gpio_request(gpio_pin, "sysfs");
    gpio_direction_input(gpio_pin);
    gpio_set_debounce(gpio_pin, 30);
    gpio_export(gpio_pin, false);

    irq_num = gpio_to_irq(gpio_pin);

    result = request_irq(irq_num,
                         (irq_handler_t) gpio_isr,
                         IRQF_TRIGGER_RISING,
                         "gpio17_interrupt",
                         NULL);

    if (result < 0) {
        printk(KERN_ERR "GPIO IRQ request failed\n");
        return result;
    }

    printk(KERN_INFO "gpio17_interrupt_driver loaded\n");
    return 0;
}

static void __exit gpio_isr_exit(void) {
    free_irq(irq_num, NULL);
    gpio_unexport(gpio_pin);
    gpio_free(gpio_pin);
    printk(KERN_INFO "gpio17_interrupt_driver unloaded\n");
}

static ssize_t register_user_pid(struct file *file, const char __user *buffer, size_t count, loff_t *pos) {
    pid_t pid;
    struct pid *pid_struct;

    if (kstrtoint_from_user(buffer, count, 10, &pid) != 0) {
        printk(KERN_ERR "Failed to convert user-space PID\n");
        return -EFAULT;
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

static const struct file_operations fops = {
    .write = register_user_pid,
};

module_init(gpio_isr_init);
module_exit(gpio_isr_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("GPIO 17 interrupt with user-space signal");
MODULE_AUTHOR("Alec Fessler");
