/* QEMU netduinoplus2, which models an STM32F405: 1 MB flash at 0x08000000.
 *
 * FLASH is the part's real length. What the image may actually spend is gated in
 * ci/budgets.toml, where a raise is a reviewed number in a diff; a linker length can only
 * ever say "does not fit".
 *
 * RAM stays at 64K, half the part's 128K, on purpose: it is what makes the learned-policy
 * check mean something. Read the weights out of flash where they lie and the run fits with
 * room over; copy them into RAM first and it does not.
 */

MEMORY
{
  FLASH : ORIGIN = 0x08000000, LENGTH = 1024K
  RAM   : ORIGIN = 0x20000000, LENGTH = 64K
}
