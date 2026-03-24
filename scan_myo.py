import asyncio
from bleak import BleakScanner

MYO_SERVICE_UUID = "d5060001-a904-deb9-4748-2c7f4a124842"

async def main():
    print("Start scanning BLE...")

    scanner = BleakScanner()
    await scanner.start()
    await asyncio.sleep(8.0)
    await scanner.stop()

    found = False

    for device, adv in scanner.discovered_devices_and_advertisement_data.values():
        name = device.name or adv.local_name or "<unknown>"
        service_uuids = [u.lower() for u in (adv.service_uuids or [])]

        print(f"Seen: {name} | {device.address} | UUIDs={service_uuids}")

        if MYO_SERVICE_UUID.lower() in service_uuids or "myo" in name.lower():
            found = True
            print("\nPossible Myo found:")
            print(f"  Name : {name}")
            print(f"  Addr : {device.address}")
            print(f"  UUIDs: {service_uuids}")

    if not found:
        print("\nMyo not found.")

if __name__ == "__main__":
    asyncio.run(main())