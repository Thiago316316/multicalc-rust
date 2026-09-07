/* QEMU virt DRAM; -bios none loads the kernel here.
 *
 * One region holds code and data both, so it is the sum that has to fit. This length is a
 * budget we set, not a limit the machine has — virt comes up with 128 MiB and the runner
 * passes no -m. It was 256K until the learned policy's 23 KB of read-only data left the
 * other checks no headroom.
 */
MEMORY
{
	RAM : ORIGIN = 0x80000000, LENGTH = 512K
}

REGION_ALIAS("REGION_TEXT", RAM);
REGION_ALIAS("REGION_RODATA", RAM);
REGION_ALIAS("REGION_DATA", RAM);
REGION_ALIAS("REGION_BSS", RAM);
REGION_ALIAS("REGION_HEAP", RAM);
REGION_ALIAS("REGION_STACK", RAM);
